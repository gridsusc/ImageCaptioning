from PIL import Image
import numpy as np,h5py, json, os, time, glob, nltk
from nltk.tokenize import RegexpTokenizer

# You might have to uncomment this line if you are running this code for the first time
# nltk.download('popular')

start = time.time()
images_path = "./Images/*"
all_images = glob.glob(images_path)
# all_images = all_images[:100]
# print(all_images[0])

all_images_list = []
for image_path in all_images:
	image = Image.open(image_path).convert('RGB')
	image = np.asarray(image.resize((256,256)))
	image = np.expand_dims(image,0)
	all_images_list.append(image)
all_images_matrix = np.concatenate(all_images_list,axis=0)
print(all_images_matrix.shape)

with h5py.File('first.hdf5', 'w') as f:
	f.create_dataset("images",data = all_images_matrix)

with open("captions.txt","r") as f:
	caption_list = f.readlines()

# Key is image name and value is list of all 5 captions for that image (Assumption here is that all images are .jpg formats)
caption_dict = {}
for i in range(1,len(caption_list)):
	im,c = caption_list[i].split('.jpg,')
	c = c.strip()
	im = im+".jpg"
	temp = caption_dict.get(im,[])
	temp.extend([c])
	caption_dict[im] = temp

### What do we do with extra spaces and punctuations in the captions. For now they have been filtered but we need to think about it later!
### Do we have to convert all characters into lower or upper case while making our vocab?

# Getting a JSON with all image captions in the same order as the hdf5 file
all_captions_list = []
for image_path in all_images:
	image_name = os.path.basename(image_path)
	all_captions_list.extend(caption_dict[image_name])
# print(all_captions_list)
with open("json_with_actual_captions.json","w") as f:
	json.dump(all_captions_list,f)

# Getting a JSON with all caption lengths in the same order as the hdf5 file
def get_sentence_len(sent):
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(sent)
	return len(tokens)

all_caption_lens = [get_sentence_len(sent) for sent in all_captions_list]
with open("json_with_caption_lens.json","w") as f:
	json.dump(all_caption_lens,f)

print(time.time()-start)