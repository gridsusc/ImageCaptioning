# ImageCaptioning

Based on this Tutorial https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning (Code was not duplicated and only referred to when there struck in a roadblock for a long time)

Update 2022/03/31 - Decided to use flickr8k dataset initially. Worked on creating the hdf5 file code and the 2 JSON files. Had a few questions in mind which were put in the python script as comments. Did not yet consider train-val-test splits and yet to make the word2idx mapping. To be able to run the code in make_dataset.py be sure to downlaod the flickr8k dataset in the required format.

Update 2022/04/01 - Token captions outputted to json, added padding, added word to index mapping, finalized image pre-processing,
decided on train test val splits following Andrej Karpathy splits.

Update 2022/04/02 - Factored all previous progress in the .py file, gave finishing touches for the train test val splits, added comments in the .py file to make it more readble

Update 2022/04/03 - Added pytorch dataset class file, In this the caption was also processed to make it a single tensor

Update 2022/05/09 - Fixed a small bug in make_dataset.py (off-by-one error) , Completed the file all_models.py which has the encoder model, attention network and decoder with attention network (cannot check for correctness now and there are bound to be bugs in this in the future which will pop up while writing the training code)

Update 2022/06/12 - Added all the training and validation code in train.py and fixed whatever was wrong in the previous files. Used black autoformatting for all files to make the code more readble. Still need to monitor the progress of the training of model as the loss does not seem to converge.  