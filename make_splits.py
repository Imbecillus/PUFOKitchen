# This script takes a data folder and creates two subsets out of it, making a 70:30 train/test-split.
import sys
import os
import random
import shutil

datafolder = sys.argv[1]
if len(sys.argv) > 2:
    ignore = sys.argv[2:]
else:
    ignore = []

# Delete old split directories and create new ones
datafolder_train = f"{datafolder}_train"
if os.path.isdir(datafolder_train):
    print("Removing old train split")
    shutil.rmtree(datafolder_train)
os.mkdir(datafolder_train)

datafolder_test = f"{datafolder}_test"
if os.path.isdir(datafolder_test):
    print("Removing old test split")
    shutil.rmtree(datafolder_test)
os.mkdir(datafolder_test)

# Obtain list of classes / subdirectories
classes = [o for o in os.listdir(datafolder)
           if os.path.isdir(os.path.join('.', datafolder, o))]

# Make selections and copy files
for c in classes:
    print(f"{c}...", end='')
    if c in ignore:
        print(' ignore')
        continue

    # Get all files
    files = [f for f in os.listdir(os.path.join(datafolder, c))
             if os.path.isfile(os.path.join(datafolder, c, f))]
    
    # Make folders for class
    os.mkdir(os.path.join(datafolder_train, c))
    os.mkdir(os.path.join(datafolder_test, c))

    # Get random selection of files
    num_train = int(0.7 * len(files))
    random.shuffle(files)
    trainfiles = files[0:num_train]

    # Copy files
    for f in files:
        f_origin = os.path.join(datafolder, c, f)
        if f in trainfiles:
            f_target = os.path.join(datafolder_train, c, f)
        else:
            f_target = os.path.join(datafolder_test, c, f)
        shutil.copyfile(f_origin, f_target)

    print(' done')

print('Finished.')