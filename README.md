# ImageCaptioning
Implementation of deep learning model that generates descriptive caption for an image

Based on this Tutorial https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

Update 2022/03/31 - Decided to use flickr8k dataset initially. Worked on creating the hdf5 file code and the 2 JSON files. Had a few questions in mind which were put in the python script as comments. Did not yet consider train-val-test splits and yet to make the word2idx mapping. To be able to run the code in make_dataset.py be sure to downlaod the flickr8k dataset in the required format.

Update 2022/04/01 - Token captions outputted to json, added padding, added word to index mapping, finalized image pre-processing,
decided on train test val splits following Andrej Karpathy splits. 