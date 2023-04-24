import json
import os
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

def load_image(im_path):
    img = Image.open(im_path)
    tensor_img = transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.Resize([256, 256],antialias=True),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
    img = tensor_img(img)
    return img

def inference_plain(im_path, saved_model_path, save_caption_file=False):
    # device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(f"For plain, using device = {device}")
    max_caption_size = 25
    state_dict = torch.load(saved_model_path, map_location=device)
    enc = state_dict["encoder"].to(device)
    dec = state_dict["decoder"].to(device)

    image = load_image(im_path)
    image = image.unsqueeze(0).to(device)
    enc_image = enc(image)
    enc_image = enc_image.view(enc_image.size(0),-1, 2048)
    prev_word = "<sos>"
    hidden_state = dec.initialize_hidden_state(enc_image.mean(dim=1))
    cell_state = dec.initialize_cell_state(enc_image.mean(dim=1))

    caption = []
    while (prev_word!="<eos>") and (len(caption)<max_caption_size):
        alphas_for_each_pixel = dec.attention_net(enc_image,hidden_state)
        gating_scalar = dec.layer_to_get_gating_scalar(hidden_state)
        gating_scalar = dec.sigmoid_act(gating_scalar)
        encoding_with_attention = gating_scalar * ((enc_image* alphas_for_each_pixel.unsqueeze(2)).sum(dim=1))
        embedding_value = dec.embedding_layer(torch.LongTensor([word2idx[prev_word]]).to(device))
        hidden_state, cell_state = dec.lstm_cell(torch.cat([embedding_value,encoding_with_attention,],dim=1,),(hidden_state,cell_state))
        scores = dec.layer_to_get_word_scores(hidden_state)
        scores = (F.softmax(scores,dim=1).squeeze().tolist())
        max_ind = max(range(len(scores)), key = lambda a:scores[a])
        prev_word = idx2word[max_ind]
        if prev_word!="<eos>":
            caption.append(prev_word)
    
    caption = " ".join(caption)

    if save_caption_file:
        im_path_dir = os.path.dirname(im_path)
        im_path_file = os.path.basename(im_path)
        caption_path_basename = im_path_file.split(".")[0] + ".txt"
        caption_path = os.path.join(im_path_dir, caption_path_basename)
        with open(caption_path, "w") as f:
            f.write(f"{im_path}:\n")
            f.write(caption)

    return caption

def inference_beam_search(im_path, saved_model_path, beam_size = 5, save_caption_file=False):
    max_caption_size = 25
    # device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    device = 'cpu'
    print(f"For beam search, using device = {device}")
    state_dict = torch.load(saved_model_path, map_location=device)
    enc = state_dict["encoder"].to(device)
    dec = state_dict["decoder"].to(device)

    image = load_image(im_path)
    image = image.unsqueeze(0).to(device)
    enc_image = enc(image)
    enc_image = enc_image.view(enc_image.size(0),-1, 2048)
    n_pixels = enc_image.shape[1]
    enc_image = enc_image.expand(beam_size, n_pixels, 2048)

    caption_prev_words = torch.LongTensor([[word2idx['<sos>']]] * beam_size).to(device)
    captions = caption_prev_words
    top_caption_scores = torch.zeros(beam_size, 1).to(device)
    # captions_alpha = torch.ones(beam_size, 1, enc.encoder_image_size, enc.encoder_image_size).to(device)
    captions_alpha = torch.ones(beam_size, 1, 16, 16).to(device)

    generated_captions, generated_caption_scores = [], []

    time_step = 0
    hidden_state = dec.initialize_hidden_state(enc_image.mean(dim=1))
    cell_state = dec.initialize_cell_state(enc_image.mean(dim=1))

    while time_step<max_caption_size:
        embedding_value = dec.embedding_layer(caption_prev_words).squeeze(1).to(device)
        alphas_for_each_pixel = dec.attention_net(enc_image, hidden_state)
        gating_scalar = dec.layer_to_get_gating_scalar(hidden_state)
        gating_scalar = dec.sigmoid_act(gating_scalar)
        encoding_with_attention = gating_scalar * ((enc_image* alphas_for_each_pixel.unsqueeze(2)).sum(dim=1))
        # alphas_for_each_pixel = alphas_for_each_pixel.view(-1, enc.encoder_image_size, enc.encoder_image_size)
        alphas_for_each_pixel = alphas_for_each_pixel.view(-1, 16, 16)
        hidden_state, cell_state = dec.lstm_cell(torch.cat([embedding_value,encoding_with_attention,],dim=1,),(hidden_state,cell_state))
        scores = dec.layer_to_get_word_scores(hidden_state)
        scores = F.softmax(scores,dim=1)
        scores = top_caption_scores.expand(scores.shape) + scores

        if time_step==0:
            top_caption_scores, top_caption_words = scores[0].topk(k=beam_size, dim=0, largest=True, sorted=True)
        else:
            top_caption_scores, top_caption_words = scores.view(-1).topk(k=beam_size, dim=0, largest=True, sorted=True)
        prev_word_inds = top_caption_words // one_hot_size
        next_word_inds = top_caption_words % one_hot_size

        captions = torch.cat([captions[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
        captions_alpha = torch.cat([captions_alpha[prev_word_inds], alphas_for_each_pixel[prev_word_inds].unsqueeze(1)], dim=1)

        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word2idx['<eos>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
        incomplete_inds, complete_inds = [],[]
        for ind, next_word in enumerate(next_word_inds):
            if next_word != word2idx['<eos>']:
                incomplete_inds.append(ind)
            else:
                complete_inds.append(ind)

        if len(complete_inds) > 0:
            generated_captions.extend(captions[complete_inds].tolist())
            generated_caption_scores.extend(top_caption_scores[complete_inds])
        beam_size -= len(complete_inds)

        captions = captions[incomplete_inds]
        captions_alpha = captions_alpha[incomplete_inds]
        hidden_state = hidden_state[prev_word_inds[incomplete_inds]]
        cell_state = cell_state[prev_word_inds[incomplete_inds]]
        enc_image = enc_image[prev_word_inds[incomplete_inds]]
        top_caption_scores = top_caption_scores[incomplete_inds].unsqueeze(1)
        caption_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        time_step += 1

    golden_ind = generated_caption_scores.index(max(generated_caption_scores))
    caption = generated_captions[golden_ind]
    caption = " ".join([idx2word[token] for token in caption])
    caption = caption.lstrip("<sos>").rstrip("<eos>")

    return caption
            

if __name__=="__main__":
    fraction = 1
    lr = 4e-4  # Learning rate
    dropout_probab = 0.5

    dataset_path_suffix = f"dataset_{fraction}" if fraction!=1 else "dataset"
    dataset_path = os.path.join(os.path.dirname(__file__), dataset_path_suffix)
    
    with open(os.path.join(dataset_path, "index_to_word.json"), "r") as f:
        idx2word = json.load(f)
    idx2word = {int(k):v for k,v in idx2word.items()}
    with open(os.path.join(dataset_path, "word_to_index_map.json"), "r") as f:
        word2idx = json.load(f)
    one_hot_size = len(word2idx)
    
    save_dir = "model_files"
    ckpt_filedir = os.path.join(os.path.dirname(__file__),save_dir)
    ckpt_filename = f"best_model_{lr}LR_{dropout_probab}dropout_{dataset_path_suffix}.pth.tar"
    ckpt_filepath = os.path.join(ckpt_filedir,ckpt_filename)
    # ckpt_filepath = "model_files_backup/best_model_0.5dropout_dataset.pth.tar"
    
    image_path = "sample2.jpg"
    caption = inference_plain(im_path = image_path, saved_model_path = ckpt_filepath, save_caption_file = False)
    print("Without beam search:", caption)
    caption = inference_beam_search(im_path = image_path, saved_model_path = ckpt_filepath, save_caption_file = False)
    print("With beam search:", caption)