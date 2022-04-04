import torch
from torch.utils.data import Dataset
import os,json, h5py

class fetch_data(Dataset):
	
	def __init__(self,dataset_path,split,transform=None):
		super().__init__()
		assert split in ['train', 'val', 'test']
		assert os.path.exists(dataset_path)
		self.n_captions_per_image = 5
		self.split = split
		image_dataset_name = f"image_dataset_{split}.hdf5"
		self.imgs = h5py.File(os.path.join(dataset_path,image_dataset_name),"r")
		self.imgs = self.imgs["default"]
		caption_dataset_name = f"tokenized_captions_{split}.json"
		caplens_dataset_name = f"caption_lengths_{split}.json"
		
		with open(os.path.join(dataset_path,caption_dataset_name),"r") as f:
			self.captions = json.load(f)
		
		with open(os.path.join(dataset_path,caplens_dataset_name),"r") as f:
			self.caplens = json.load(f)
		
		self.transform = transform
		
		word2idx_map_name = "word_to_index_map.json"
		with open(os.path.join(dataset_path,word2idx_map_name),"r") as f:
			self.word2idx = json.load(f)
	
	def __getitem__(self,i):
		img = self.imgs[i//self.n_captions_per_image]
		img = torch.from_numpy(img)
		if self.transform!=None:
			img = self.transform(img)
		caption = self.captions[i]
		caplen = self.caplens[i]
		def get_one_hot_encoding(idx,dim):
			res = [0]*dim
			res[idx]=1
			return res
		dim = len(self.word2idx)
		caption = torch.FloatTensor([get_one_hot_encoding(self.word2idx.get(word,self.word2idx["<unk>"]),dim) for word in caption])
		if self.split=="train":
			return img,caption,caplen
		else:
			all_captions_for_image = []
			start = ((i//self.n_captions_per_image)*self.n_captions_per_image)
			end = ((i//self.n_captions_per_image)*self.n_captions_per_image)+self.n_captions_per_image
			for i in range(start,end):
				all_captions_for_image.append(torch.FloatTensor([get_one_hot_encoding(self.word2idx.get(word,self.word2idx["<unk>"]),dim) for word in self.captions[i]]))
			return img,caption,caplen,all_captions_for_image
	
	def __len__(self):
		return len(self.captions)