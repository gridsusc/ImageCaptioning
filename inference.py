import json
import os
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from all_models import encoder

dataset_path_suffix = "dataset"
dataset_path = os.path.join(os.path.dirname(__file__), dataset_path_suffix)
with open(os.path.join(dataset_path, "index_to_word.json"), "r") as f:
    idx2word = json.load(f)
idx2word = {int(k):v for k,v in idx2word.items()}
with open(os.path.join(dataset_path, "word_to_index_map.json"), "r") as f:
    word2idx = json.load(f)

def load_image(im_path):
    img = Image.open(im_path)
    tensor_img = transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.Resize([256, 256],antialias=True),
                transforms.ConvertImageDtype(torch.float),
            ]
        )
    img = tensor_img(img)
    return img

def inference(im_path, saved_model_path, save_caption_file=False):
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(saved_model_path)
    enc = state_dict["encoder"]
    dec = state_dict["decoder"]

    image = load_image(im_path)
    image = image.unsqueeze(0)
    enc_image = enc(image)
    enc_image = enc_image.view(enc_image.size(0),-1, 2048)
    n_pixels = enc_image.size(dim=1)

    prev_word = "<sos>"
    hidden_state = dec.initialize_hidden_state(enc_image.mean(dim=1))
    cell_state = dec.initialize_cell_state(enc_image.mean(dim=1))

    caption = []
    while (prev_word!="<eos>") and (len(caption)<25):
        alphas_for_each_pixel = dec.attention_net(enc_image,hidden_state)
        gating_scalar = dec.layer_to_get_gating_scalar(hidden_state)
        gating_scalar = dec.sigmoid_act(gating_scalar)
        encoding_with_attention = gating_scalar * ((enc_image* alphas_for_each_pixel.unsqueeze(2)).sum(dim=1))
        embedding_value = dec.embedding_layer(torch.LongTensor([word2idx[prev_word]]))
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

if __name__=="__main__":
    model_path = "model_files/best_model_0.5dropout_dataset.pth.tar"
    image_path = "sample2.jpg"
    caption = inference(im_path = image_path, saved_model_path = model_path, save_caption_file = False)
    print(caption)