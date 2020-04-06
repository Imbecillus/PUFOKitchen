# What's this?
This will become a small for-fun project that'll try to recognize food pictures by different members of the *Das Podcast-UFO Telegram Group* using PyTorch.

# Files
## dataset.py
A simple dataset that parses a root folder for subdirectories for each of the classes, and then passes the subdirectories for images.

Images are loaded, transforms applied, and then converted to PyTorch tensors in the format expected by PyTorch.

## train.py
This is were the training will happen. Currently only displays a random selection of images from the dataset folder.