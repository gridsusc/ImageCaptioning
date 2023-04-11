import os
import json
import time

import torch
from nltk.translate.bleu_score import corpus_bleu
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from all_models import *
from pytorch_dataset_class import *

with open("./dataset/word_to_index_map.json") as f:
    n_words = max(json.load(f).values())+1

dataset_path = "./dataset"
attention_size = 512
one_hot_size = n_words
embedding_size = 512
dropout_probab = 0.5

doubly_stochastic_regularization_parameter = 1

# n_epochs = 120
n_epochs = 5
no_progress_epoch_allowance = 3
bs = 24  # Batch size
lr = 1e-3  # Learning rate
pf = 64  # Print frequency

ckpt_filepath = f"./model_files/best_model_{dropout_probab}dropout.pth.tar"

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device being used is {device}")

with open(os.path.join(dataset_path, "index_to_word.json"), "r") as f:
    idx2word = json.load(f)

idx2word = {int(k): v for k, v in idx2word.items()}

train_loss_list = []


def train(data_loader, encoder, dec, optimizer, loss_fn):
    epoch_start = time.time()

    encoder.train()
    dec.train()

    avg_loss_epoch = (0, 0)

    for i, (imgs, caps, caps_len) in enumerate(data_loader):
        imgs = imgs.to(device)
        caps = caps.to(device)
        caps_len = caps_len.to(device)

        enc_imgs = encoder(imgs)
        (
            predictions,
            all_alphas,
            sorted_enc_captions,
            sorted_caps_len,
            sorted_indices_list,
        ) = dec(enc_imgs, caps, caps_len)

        to_be_predicted = sorted_enc_captions[:, 1:]  ### To remove the <sos> token

        caps_len_minus1 = list(
            map(lambda a: a - 1, sorted_caps_len)
        )  ### As <eos> token was not decoded

        predictions = torch.nn.utils.rnn.pack_padded_sequence(
            predictions, caps_len_minus1, batch_first=True
        )
        to_be_predicted = torch.nn.utils.rnn.pack_padded_sequence(
            to_be_predicted, caps_len_minus1, batch_first=True
        )

        predictions = predictions.data
        to_be_predicted = to_be_predicted.data

        loss = loss_fn(predictions, to_be_predicted)

        loss += (
            doubly_stochastic_regularization_parameter
            * ((1 - all_alphas.sum(dim=1)) ** 2).mean()
        )

        avg_loss_epoch = (
            avg_loss_epoch[0] + (loss * sum(caps_len_minus1)),
            avg_loss_epoch[1] + sum(caps_len_minus1),
        )

        loss.backward()
        optimizer.zero_grad()
        optimizer.step()

        if i % pf == 0:
            print(
                f"For training:- Progress: {i+1}/{len(data_loader)} ; Average loss in this till now: {avg_loss_epoch[0]/avg_loss_epoch[1]} ; Elapsed time in this epoch: {time.time()-epoch_start}"
            )

    avg_loss_epoch = avg_loss_epoch[0] / avg_loss_epoch[1]
    global train_loss_list
    train_loss_list.append(avg_loss_epoch)


def validation(data_loader, encoder, decoder_with_attention_net, loss_fn):
    val_start = time.time()

    encoder.eval()
    decoder_with_attention_net.eval()
    references, hypothesis = [], []

    avg_loss_epoch = (0, 0)

    with torch.no_grad():
        for i, (imgs, caps, caps_len, all_caps) in enumerate(data_loader):
            imgs = imgs.to(device)
            caps = caps.to(device)
            caps_len = caps_len.to(device)

            enc_imgs = encoder(imgs)
            (
                predictions,
                all_alphas,
                sorted_enc_captions,
                sorted_caps_len,
                sorted_indices_list,
            ) = decoder_with_attention_net(enc_imgs, caps, caps_len)

            to_be_predicted = sorted_enc_captions[:, 1:]  ### To remove the <sos> token

            caps_len_minus1 = list(
                map(lambda a: a - 1, sorted_caps_len)
            )  ### As <eos> token was not decoded

            predictions_for_bleu = predictions.clone()

            predictions = torch.nn.utils.rnn.pack_padded_sequence(
                predictions, caps_len_minus1, batch_first=True
            )
            to_be_predicted = torch.nn.utils.rnn.pack_padded_sequence(
                to_be_predicted, caps_len_minus1, batch_first=True
            )

            predictions = predictions.data
            to_be_predicted = to_be_predicted.data

            loss = loss_fn(predictions, to_be_predicted)

            loss += (
                doubly_stochastic_regularization_parameter
                * ((1 - all_alphas.sum(dim=1)) ** 2).mean()
            )

            avg_loss_epoch = (
                avg_loss_epoch[0] + (loss * sum(caps_len_minus1)),
                avg_loss_epoch[1] + sum(caps_len_minus1),
            )

            if i % pf == 0:
                print(
                    f"For validation:- Progress: {i+1}/{len(data_loader)} ; Average loss in this till now: {avg_loss_epoch[0]/avg_loss_epoch[1]} ; Elapsed time in this epoch: {time.time()-val_start}"
                )

            # Somehow calculate BLEU score for early stopping
            # To compare pred and gt you have to remove start and pad tokens as they are not available in pred

            # For references:
            sorted_all_caps = all_caps[sorted_indices_list]

            local_references = []

            for j in range(sorted_all_caps.shape[0]):
                all_five_caps = []
                for k in sorted_all_caps[j]:
                    one_cap = []
                    for x in range(k.shape[0]):
                        idx = k[x].tolist().index(1)
                        w = idx2word[idx]
                        if w != "<sos" and w != "<pad>":
                            one_cap.append(w)
                    all_five_caps.append(one_cap)
                local_references.append(all_five_caps)

            references.extend(local_references)

            # For hypothesis:

            local_hypothesis = []

            preds_indices = torch.argmax(predictions_for_bleu, dim=2)
            for one in preds_indices:
                one_cap = []
                for w in one:
                    w = idx2word[w.item()]
                    if w != "<sos>" and w != "<pad>":
                        one_cap.append(w)
                local_hypothesis.append(one_cap)
            hypothesis.extend(local_hypothesis)

    bleuscore = corpus_bleu(references, hypothesis)

    print(
        f"For validation:- Average loss : {avg_loss_epoch[0]/avg_loss_epoch[1]} ; Elapsed time in this epoch: {time.time()-val_start} ; BLEU score : {bleuscore}"
    )
    return bleuscore


enc = encoder().to(device)
dec = decoder_with_attention_net(
    attention_size, one_hot_size, embedding_size, dropout_probab, device
).to(device)
# decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, dec.parameters()),lr=lr)
optimizer = torch.optim.Adam(dec.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss().to(device)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([normalize])
train_intermediate = fetch_data("./dataset", "train", transform=transform)
val_intermediate = fetch_data("./dataset", "val", transform=transform)
train_loader = DataLoader(train_intermediate, batch_size=bs, shuffle=True)
val_loader = DataLoader(val_intermediate, batch_size=bs, shuffle=True)

no_progress_epochs = 0
max_bleu = 0
for i in range(n_epochs):
    if no_progress_epochs == no_progress_epoch_allowance:
        break
    print(f"Started training for epoch {i+1}")
    train(train_loader, enc, dec, optimizer, loss_fn)
    print(f"Started Validation for epoch {i+1}")
    new_bleu = validation(val_loader, enc, dec, loss_fn)
    if new_bleu > max_bleu:
        max_bleu = new_bleu
        no_progress_epochs = 0
        state_dict = {}
        state_dict["val_bleuscore"] = new_bleu
        state_dict["encoder"] = enc
        state_dict["decoder"] = dec
        state_dict["optimizer"] = optimizer
        torch.save(state_dict, ckpt_filepath)
    else:
        no_progress_epochs += 1
        print(f"No progress for {no_progress_epochs} epochs")
