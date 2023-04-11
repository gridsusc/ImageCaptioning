from PIL import Image
import numpy as np
import h5py, json, os, time
from nltk.tokenize import RegexpTokenizer
import torch
from torchvision import transforms

# You can download dataset from: 
# https://www.kaggle.com/datasets/adityajn105/flickr8k for images and captions
# https://www.kaggle.com/datasets/shtvkumar/karpathy-splits?resource=download&select=dataset_flickr8k.json for splits

# You might have to uncomment these lines if you are running this code for the first time (to downlaod the required nltk tokenizer)
# import nltk
# nltk.download('popular')

start = time.time()

### Defining all the file paths and creating our dataset_path if it does not exist already
images_path = "./Images/"
split_path = "./caption_datasets/"
dataset_path = "./dataset/"
os.makedirs(dataset_path, exist_ok=True)

### Getting the ordered list of all image names in all splits
split_filename = "dataset_flickr8k.json" ### 
with open(os.path.join(split_path, split_filename), "r") as f:
    cap_data = json.load(f)
train, test, val = [], [], []
file_split = map(lambda a: (a["filename"], a["split"]), cap_data["images"])
for file, split in file_split:
    if split == "train":
        train.append(file)
    if split == "test":
        test.append(file)
    if split == "val":
        val.append(file)

train,test,val = train[:60],test[:10],val[:10]

print(
    f"Number of images in:\ntrain = {len(train)}\ntest = {len(test)}\nval = {len(val)}\n"
)

### Making HDF5 files for all splits in the same order as above
### (All HDF5 files have images in pytorch format (n_images,c,h,w) with all values in the float range of [0,1])
def createHDF5(images_path, names, split, dataset_path):
    image_tensor_list = []
    for name in names:
        path = os.path.join(images_path, name)
        img = Image.open(path)

        tensor_img = transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.Resize([256, 256]),
                transforms.ConvertImageDtype(torch.float),
            ]
        )
        image_tensor_list.append(tensor_img(img))
    images = torch.stack(image_tensor_list)
    filename = f"image_dataset_{split}.hdf5"
    with h5py.File(os.path.join(dataset_path, filename), "w") as f:
        f.create_dataset("default", data=images)


createHDF5(images_path, train, "train", dataset_path)
createHDF5(images_path, test, "test", dataset_path)
createHDF5(images_path, val, "val", dataset_path)

### Parsing the captions file and make a dictionary where Key is image name and value is list of all 5 captions(stripped, converted to lowercase, and tokenized without punctuations)
### (Assumption here is that all images are .jpg formats)
captions_filename = "captions.txt"
with open(os.path.join(split_path, captions_filename), "r") as f:
    caption_list = f.readlines()
caption_dict = {}
tokenizer = RegexpTokenizer(r"\w+")
for i in range(1, len(caption_list)):
    im, c = caption_list[i].split(".jpg,")
    c = c.strip().lower()
    c = tokenizer.tokenize(c)
    im = im + ".jpg"
    temp = caption_dict.get(im, [])
    temp.extend([c])
    caption_dict[im] = temp

### Getting word counts from the train dataset and getting captions for all splits
train_caps_json, test_caps_json, val_caps_json = [], [], []
wordCount = {}
for name in train:
    captions_for_image = caption_dict[name]
    train_caps_json.extend(captions_for_image)
    for caption in captions_for_image:
        for word in caption:
            wordCount[word] = wordCount.get(word, 0) + 1
for name in test:
    captions_for_image = caption_dict[name]
    test_caps_json.extend(captions_for_image)
for name in val:
    captions_for_image = caption_dict[name]
    val_caps_json.extend(captions_for_image)

### Generating a vocab (based on the train data) with the word2idx and idx2word saved in JSONs
sortedWords = sorted(wordCount.items(), key=lambda a: -a[1])
vocab_size = 5000
if len(sortedWords) < vocab_size:
    vocab_size = len(sortedWords)
word_to_index = {word[0]: i for i, word in enumerate(sortedWords[:vocab_size])}
# tokens = {"<sos>": vocab_size+1, "<eos>": vocab_size+2, "<pad>": vocab_size+3, "<unk>": vocab_size+4}
tokens = {
    "<sos>": vocab_size,
    "<eos>": vocab_size + 1,
    "<pad>": vocab_size + 2,
    "<unk>": vocab_size + 3,
}  ## Had to fix this later while writing all_models.py
word_to_index.update(tokens)
index_to_word = {v: k for k, v in word_to_index.items()}
word2idx_filename = "word_to_index_map.json"
idx2word_filename = "index_to_word.json"
with open(os.path.join(dataset_path, word2idx_filename), "w") as f:
    f.write(json.dumps(word_to_index, indent =4))
with open(os.path.join(dataset_path, idx2word_filename), "w") as f:
    f.write(json.dumps(index_to_word, indent =4))

### Defining the baseCaptionLen based on max caption length for train dataset to decide the amount of padding
baseCaptionLen = max(map(len, train_caps_json)) + 2

### Saving the padded captions with appropraite padding (<pad>) and <sos> and <eos> implemented for all splits
### Saving all caption lengths (<sos> and <eos> included in lengths) for all splits
def padEachCaption(caption, maxlen):
    if len(caption) > maxlen - 2:
        caption = caption[: maxlen - 2]
    padlist = ["<pad>"] * (maxlen - 2 - len(caption))
    return ["<sos>"] + caption + ["<eos>"] + padlist


padded_train_caps_json = list(
    map(lambda a: padEachCaption(a, baseCaptionLen), train_caps_json)
)
padded_test_caps_json = list(
    map(lambda a: padEachCaption(a, baseCaptionLen), test_caps_json)
)
padded_val_caps_json = list(
    map(lambda a: padEachCaption(a, baseCaptionLen), val_caps_json)
)
length_train_caps_json = list(
    map(lambda a: min(len(a) + 2, baseCaptionLen), train_caps_json)
)
length_test_caps_json = list(
    map(lambda a: min(len(a) + 2, baseCaptionLen), test_caps_json)
)
length_val_caps_json = list(
    map(lambda a: min(len(a) + 2, baseCaptionLen), val_caps_json)
)
train_caps_filename = "tokenized_captions_train.json"
test_caps_filename = "tokenized_captions_test.json"
val_caps_filename = "tokenized_captions_val.json"
with open(os.path.join(dataset_path, train_caps_filename), "w") as f:
    f.write(json.dumps(padded_train_caps_json, indent=4))
with open(os.path.join(dataset_path, test_caps_filename), "w") as f:
    f.write(json.dumps(padded_test_caps_json, indent=4))
with open(os.path.join(dataset_path, val_caps_filename), "w") as f:
    f.write(json.dumps(padded_val_caps_json, indent=4))
train_caps_len_filename = "caption_lengths_train.json"
test_caps_len_filename = "caption_lengths_test.json"
val_caps_len_filename = "caption_lengths_val.json"
with open(os.path.join(dataset_path, train_caps_len_filename), "w") as f:
    f.write(json.dumps(length_train_caps_json, indent=4))
with open(os.path.join(dataset_path, test_caps_len_filename), "w") as f:
    f.write(json.dumps(length_test_caps_json, indent=4))
with open(os.path.join(dataset_path, val_caps_len_filename), "w") as f:
    f.write(json.dumps(length_val_caps_json, indent=4))

print(f"Duration: {time.time()-start}")
