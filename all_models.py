import torch
from torch import nn
from torchvision import models, transforms

### Assuming here that we encode our image to 14x14
class encoder(nn.Module):
    def __init__(self,encoder_image_size = 14):
        super(encoder, self).__init__()
        cnn_full = models.resnet101(weights="ResNet101_Weights.DEFAULT")
        layers = list(cnn_full.children())[:-2]
        layers.append(nn.AdaptiveAvgPool2d((encoder_image_size, encoder_image_size)))
        self.all_layers_encoder = nn.Sequential(*layers)

    def forward(self, x):
        out = self.all_layers_encoder(x)
        out = out.permute(0, 2, 3, 1)
        return out


# from pytorch_dataset_class import fetch_data
# from torchsummary import summary
# from torch.utils.data import DataLoader
# enc = encoder()
# summary(enc,(3,256,256))
# exit()
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# transform = transforms.Compose([normalize])
# train_data = fetch_data("./dataset","train",transform=transform)
# train_data_batches = DataLoader(train_data,64,shuffle=True)

# for batch in train_data_batches:
# 	print(batch[0].shape)
# 	passed_through_enc = enc(batch[0])
# 	print(passed_through_enc.shape)
# 	exit()


class attention_net(nn.Module):
    def __init__(self, attention_size):
        super(attention_net, self).__init__()
        self.layer_for_encoded_image = nn.Linear(2048, attention_size)
        self.layer_for_decoder = nn.Linear(512, attention_size)
        self.final_linear_layer = nn.Linear(attention_size, 1)
        self.soft = nn.Softmax(dim=1)
        self.act = nn.ReLU()

    def forward(self, enc_image, decoder_out):
        """
        enc_image should be (batches,14*14,2048) and decoder_out should be (batches,512)
        enc_image is taken to (batches,14*14,attention_size) and decoder_out is also taken to (batches,attention_size)
        decoder_out is taken to (batches,1,attention_size) and added with the new_enc_image (adding decoder_out to each pixel based on broadcasting principles)
        The addition is of size (batches,14*14,attention_size) which is now taken to (batches,14*14,1)
        The final above output is now reshaped to (batches,14*14) which now corresponds to the actual alphas for each pixel
        """

        image_for_attention = self.layer_for_encoded_image(enc_image)
        decoder_out_for_attention = self.layer_for_decoder(decoder_out)

        # Add decoder_out_for_attention to every pixel
        added_for_final_linear_layer = self.act(
            image_for_attention + decoder_out_for_attention.unsqueeze(1)
        ).squeeze(2)

        passed_through_final_layer = self.final_linear_layer(
            added_for_final_linear_layer
        )
        alphas_for_each_pixel = self.soft(passed_through_final_layer)
        # alphas_for_each_pixel should be (batches,14*14)

        return alphas_for_each_pixel


class decoder_with_attention_net(nn.Module):
    def __init__(
        self, attention_size, one_hot_size, embedding_size, dropout_probab, device
    ):
        super(decoder_with_attention_net, self).__init__()
        self.one_hot_size = one_hot_size
        self.attention_size = attention_size
        self.embedding_size = embedding_size
        self.dropout_layer = nn.Dropout(p=dropout_probab)
        self.attention_net = attention_net(attention_size)
        self.embedding_layer = nn.Embedding(one_hot_size, embedding_size)
        self.lstm_cell = nn.LSTMCell(embedding_size + 2048, 512)
        self.initialize_hidden_state = nn.Linear(2048, 512)
        self.initialize_cell_state = nn.Linear(2048, 512)
        self.layer_to_get_gating_scalar = nn.Linear(512, 2048)
        self.sigmoid_act = nn.Sigmoid()
        self.layer_to_get_word_scores = nn.Linear(512, one_hot_size)
        self.device = device

    def forward(self, enc_image, enc_captions, caps_len):
        bs = enc_image.size(dim=0)
        enc_image = enc_image.view(
            bs, -1, 2048
        )  # Making the enc_image (batches,14*14,2048)
        n_pixels = enc_image.size(dim=1)
        sorted_indices_list = sorted(
            list(range(len(caps_len))), key=lambda a: -caps_len[a]
        )
        caps_len = [caps_len[i].item() for i in sorted_indices_list]
        sorted_enc_image = enc_image[sorted_indices_list]
        sorted_enc_captions = enc_captions[sorted_indices_list].long()
        sorted_enc_captions = torch.argmax(sorted_enc_captions, dim=2)
        embedding_values = self.embedding_layer(
            sorted_enc_captions
        )  # Probably not expecting one-hot (maybe just the index)
        hidden_state = self.initialize_hidden_state(sorted_enc_image.mean(dim=1))
        cell_state = self.initialize_cell_state(sorted_enc_image.mean(dim=1))

        biggest_caption = max(caps_len)

        predictions = torch.zeros(bs, biggest_caption - 1, self.one_hot_size).to(
            self.device
        )
        all_alphas = torch.zeros(bs, biggest_caption - 1, n_pixels).to(self.device)

        for i in range(
            biggest_caption - 1
        ):  ## To avoid the last <eos> token being decoded
            n_samples_to_be_processed = 0
            for el in caps_len:
                if (el - 1) > i:  ## el-1 because we had to get rid of <eos> token
                    n_samples_to_be_processed += 1

            alphas_for_each_pixel = self.attention_net(
                sorted_enc_image[:n_samples_to_be_processed],
                hidden_state[:n_samples_to_be_processed],
            )
            gating_scalar = self.layer_to_get_gating_scalar(
                hidden_state[:n_samples_to_be_processed]
            )
            gating_scalar = self.sigmoid_act(gating_scalar)
            encoding_with_attention = gating_scalar * (
                (
                    sorted_enc_image[:n_samples_to_be_processed]
                    * alphas_for_each_pixel
                ).sum(dim=1)
            )
            hidden_state, cell_state = self.lstm_cell(
                torch.cat(
                    [
                        embedding_values[:n_samples_to_be_processed, i, :],
                        encoding_with_attention,
                    ],
                    dim=1,
                ),
                (
                    hidden_state[:n_samples_to_be_processed],
                    cell_state[:n_samples_to_be_processed],
                ),
            )
            scores = self.layer_to_get_word_scores(self.dropout_layer(hidden_state))
            predictions[:n_samples_to_be_processed, i, :] = scores
            all_alphas[:n_samples_to_be_processed, i, :] = alphas_for_each_pixel.squeeze(2)

        return (
            predictions,
            all_alphas,
            sorted_enc_captions,
            caps_len,
            sorted_indices_list,
        )
