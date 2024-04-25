


####################### IMPORTS ##############################################
##############################################################################
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pdb

import sklearn  # scikit-learn
import torch
from PIL import Image

# import pytorch modules
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
print("Fin des imports")
print("_______________")

####################### FUNCTIONS ############################################
##############################################################################

# Helper function: we monitor the accuracy during the training
def prediction_accuracy(predict,labels):
  accuracy = (predict == labels).sum()/(labels.shape[0])
  return accuracy

def make_labels(labels):
  l_labels = []
  for i in range(len(labels)):
    l_labels.append([])
    for j in range(8):
      if labels[i] == j:
        l_labels[-1].append(1)
      else:
        l_labels[-1].append(0)
  return l_labels



####################### MAIN #################################################
##############################################################################
n_images_train = 200 # max 18998
n_images_test = 50 # max 6333
desired_size = (512, 384) # The dimensions of the images in the dataset

### Importer les données d'entrainement et les données de test
X_train = [] # Create an empty list to store the images
X_test = [] # Create an empty list to store the images
#region : remplir le vecteur X_train avec les images d'entrainement
image_files_train = os.listdir('Train\Train') # List all the files in the directory
print("Loading training images...")
tiny_train_image_files = image_files_train[:n_images_train] # Take the first 10 files

for image_file in tqdm(tiny_train_image_files): # Loop over the files
    image = Image.open('Train\Train\\' + image_file)
    image = image.resize(desired_size)
    image_np = np.array(image)
    X_train.append(image_np)
#endregion
#region : remplir le vecteur X_test avec les images de test
image_files_test = os.listdir('Test\Test')
print("Loading test images...")
tiny_test_image_files = image_files_test[:n_images_test]
for image_file in tqdm(tiny_test_image_files):
    image = Image.open('Test\Test\\' + image_file)
    image = image.resize(desired_size)  # Resize the image
    image_np = np.array(image)
    X_test.append(image_np)
#endregion
Y_train = [] # Create an empty list to store the labels of training data
Y_test = [] # Create an empty list to store the labels of test data
#region : remplir les vecteur Y_train et Y_test
print("Loading training and test labels...")
data_train = pd.read_csv('metadataTrain.csv')
tiny_data_train = data_train[:n_images_train]
Y_train = tiny_data_train['CLASS'].values        # numpyarray of integers
data_test = pd.read_csv('SampleSubmission.csv')
tiny_data_test = data_test[:n_images_test]
Y_test = tiny_data_test['CLASS'].values
#endregion

X_train = np.array(X_train)
X_test = np.array(X_test)


print("Fin des remplissages des vecteurs")
print("_________________________________")

X_train = np.array(X_train).reshape(len(X_train), -1)
X_test = np.array(X_test).reshape(len(X_test), -1)

training_dataset = TensorDataset(torch.tensor(X_train).float(),torch.unsqueeze(torch.tensor(Y_train),-1).float()) # Creates a pytorch Dataset object
test_dataset = TensorDataset(torch.tensor(X_test).float(),torch.unsqueeze(torch.tensor(Y_test),-1).float()) # Creates a pytorch Dataset object
# so the wall purpose of a dataset is the get_item(i) method, which returns the i-th element of the dataset as a torch tensor
# such dataset are just much easier to manipulate, we'll see why later


### Using GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


### Créer le modèle du Programme de Machine Learning
d_im = desired_size[0]*desired_size[1]*3 # 3 channels for the RGB
# d_im = 3
n_1l = 500
n_2l = 200
n_3l = 100
n_out = 8    # 4 classes as an output

## Let's first try with the sequential API
mon_model = nn.Sequential()
mon_model.add_module('1st_linear_layer', nn.Linear(d_im,n_1l))
mon_model.add_module('relue1', nn.ReLU())
mon_model.add_module('2nd_linear_layer', nn.Linear(n_1l,n_2l))
mon_model.add_module('relue2', nn.ReLU())
mon_model.add_module('3rd_linear_layer', nn.Linear(n_2l,n_3l))
mon_model.add_module('relue', nn.ReLU())
mon_model.add_module('final_linear_layer', nn.Linear(n_3l,n_out))
# mon_model.add_module('softmax', nn.Softmax(dim=1))
mon_model = mon_model.to(device)

learning_rate = 0.01
n_epochs = 12
batch_size = 10

## Binary Cross Entropy loss from PyTorch
criterion = nn.CrossEntropyLoss(reduction='mean')

optimizer = optim.Adam(mon_model.parameters(), lr = learning_rate)

training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, drop_last=False) # apparently this modifies the dataset to create minibatches inside of it.
test_loader     = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
print("Fin de la création du modèle et des variables nécessaires")
print("_________________________________________________________")




####### Entrainement du NN ####################
mon_model.train() # put model in training mode (required as some layers, for instance dropout, have different behavior in training and evaluation mode)

for epoch in range(0,n_epochs): # n_epochs is the number of training iterations
  # Initialize the loss storage to plot it later
  train_loss=0.0
  all_labels = []
  all_predicted = []

  with tqdm(training_loader, unit="batch") as tepoch: # tepoch is the variable that hold each batch of data as we iterate over training_loader
    for data, labels in tepoch:
      tepoch.set_description(f"Epoch {epoch}")

      # Put the data on device
      data = data.to(device)
      labels = labels.to(device).squeeze()
      labels = make_labels(labels)
      labels = torch.tensor(labels).float()
      #print(type(labels))
      #print(labels)

      # 1. Do the forward pass : 
      y_predict = mon_model(data)  # performs the forward pass, which computes the predicted outputs by passing inputs to the model.
      # it gets better and better as the training goes on
      print(f"y_predict : {y_predict}")
      print(f"labels : {labels}")
      # 2. Compute the loss :
      # y_predict = torch.argmax(y_predict, dim=1).float() # TODO : this is where it blocks
      loss = criterion(y_predict, labels)  # computes the loss between the predicted and actual labels. And then do the sum I guess
      print(f"loss : {loss}")
      # 3. Reset gradients to 0 : optimizer = optim.Adam(model.parameters(), lr = learning_rate)
      optimizer.zero_grad() # resets the gradients to zero before starting to do backprojections because PyTorch accumulates the gradients on subsequent backward passes.

      # 4. Do the backward pass : 
      loss.backward() # computes the derivative of the loss % to parameters (or anything requiring gradients) using backpropagation.

      # 5. Call the parameter update
      optimizer.step() # causes the optimizer to take a step based on the gradients of the parameters.

      # Compute the loss
      train_loss += loss.item()
      # Store labels and class predictions
      all_labels.extend(labels.tolist())
      all_predicted.extend((y_predict>=0.5).tolist())

  print('Epoch {}: Train Loss: {:.4f}'.format(epoch, train_loss/len(training_loader.dataset)))
  print('Epoch {}: Train Accuracy: {:.4f}'.format(epoch, prediction_accuracy(np.array(all_predicted),np.array(all_labels))))

  

mon_model.eval()

all_predicted = []
all_labels = []

with tqdm(test_loader, unit="batch") as tepoch:
  for data, labels in tepoch:
    all_labels.extend(labels.tolist())

    data = data.to(device)
    y_predict = mon_model(data)
    all_predicted.extend((y_predict>=0.5).tolist())

test_accuracy = prediction_accuracy(np.array(all_predicted),np.array(all_labels))

print("\nTest Accuracy:", test_accuracy)
















































'''
  
The error message is indicating that the tensors in your `X_test` list have inconsistent sizes. The `torch.cat()` function is trying to concatenate
 them along dimension 0, but it's finding that the size of tensor number 3 is different from the others.

This is likely due to the images in your test set having different sizes. The `view()` function is used to flatten the image tensors, but if the images
 have different sizes to begin with, the flattened tensors will also have different sizes.

To fix this, you need to ensure that all images are the same size before they are converted to tensors and flattened. You can use the `resize()` 
function from the PIL library to resize the images. Here's how you can do it:



```python
from PIL import Image

# Define the size you want all images to be
desired_size = (224, 224)

# Then, in your loop where you load and process the images:
for image_path in image_paths:
    image = Image.open(image_path)
    image = image.resize(desired_size)  # Resize the image
    image_np = np.array(image)
    image_tensor = torch.from_numpy(image_np)
    flattened = image_tensor.view(1, -1)
    X_test.append(flattened)
```

This will ensure that all images are the same size before they are converted to tensors and flattened, which should resolve the `RuntimeError` you're seeing.

'''

