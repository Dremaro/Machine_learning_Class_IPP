import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pdb

import sklearn  # scikit-learn
import torch

# import pytorch modules
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import shutil



# original directory
original_train = 'original_data\Train\\'
original_test = 'original_data\Test\\'

# Define the directory to move the mask and images files to
train_image_folder = 'Train\Train'
train_mask_folder = 'Train\Mask'

test_image_folder = 'Test\Test'
test_mask_folder = 'Test\Mask'

# Make sure the mask directory exists
os.makedirs(train_image_folder, exist_ok=True)
os.makedirs(train_mask_folder, exist_ok=True)
os.makedirs(test_image_folder, exist_ok=True)
os.makedirs(test_mask_folder, exist_ok=True)


# Get a list of all image files in the directory
train_image_files = os.listdir(original_train)
test_image_files = os.listdir(original_test)


for image_file in tqdm(train_image_files):
    # If the file is a mask file
    if 'seg' in image_file:
        # Move the file to the mask directory
        shutil.move(os.path.join(original_train, image_file), os.path.join(train_mask_folder, image_file))
    else:
        shutil.move(os.path.join(original_train, image_file), os.path.join(train_image_folder, image_file))
for image_file in tqdm(test_image_files):
    # If the file is a mask file
    if 'seg' in image_file:
        # Move the file to the mask directory
        shutil.move(os.path.join(original_test, image_file), os.path.join(test_mask_folder, image_file))
    else:
        shutil.move(os.path.join(original_test, image_file), os.path.join(test_image_folder, image_file))

# Now get a list of all image files in the directory again, this time without the mask files
image_files = os.listdir(train_image_folder)





