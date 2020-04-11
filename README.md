# What's this?
This will become a small for-fun project that'll try to recognize food pictures by different members of the *Das Podcast-UFO Telegram Group* using PyTorch.

# Files
## dataset.py
A simple dataset that parses a root folder for subdirectories for each of the classes, and then passes the subdirectories for images.

Images are loaded, transforms applied, and then converted to PyTorch tensors in the format expected by PyTorch.

## make_splits.py
Takes the full dataset directory and creates two subsets, using a 70:30 train/test-split for each class. Selection of images within the classes is random.
Usage: python make_splits.py "path/to/datafolder" (opt: list of classes to be ignored)

## train.py
This is were the training will happen. Currently only displays a random selection of images from the dataset folder.