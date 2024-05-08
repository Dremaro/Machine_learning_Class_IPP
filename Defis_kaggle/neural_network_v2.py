


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

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
print("Fin des imports")
print("_______________")

################# FUNCTIONS AND CLASSES ######################################
##############################################################################

def vector_to_class(x):
  y = torch.argmax(F.softmax(x,dim=1),axis=1)
  return y

def prediction_accuracy(predict,labels):
  accuracy = (predict == labels).sum()/(labels.shape[0])
  return accuracy

def test_mnist_images(model,testset):
  model.eval()
  plt.figure(figsize=(10, 6))
  for idx in range(0,10):
      plt.subplot(2, 5, idx+1)
      rand_ind = np.random.randint(0,len(testset))
      plt.imshow(np.reshape(testset[rand_ind][0].detach().cpu().numpy(),(28,28)),cmap='gray')
      # get prediction
      model_prediction = np.argmax(model(torch.unsqueeze(testset[rand_ind][0], dim=0).to(device)).detach().cpu().numpy(),axis=1).squeeze()
      plt.title(str(int(model_prediction)))

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, labels_csv, transform=None, Masks=False, n_elements=None):
        self.img_dir = img_dir
        self.labels_df = pd.read_csv(labels_csv)
        if n_elements is not None:
            self.labels_df = self.labels_df.iloc[:n_elements]
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx): # permet d'accéder à une image/class en particulier du dataset
        img_path = os.path.join(self.img_dir, self.labels_df.iloc[idx, 0]+".jpg")
        image = Image.open(img_path).convert('RGB')
        # label = self.labels_df.iloc[idx, 'CLASS']
        label = self.labels_df.loc[idx, 'CLASS']
        if self.transform:
            image = self.transform(image)
        return image, label, self.labels_df.iloc[idx, 0]


####################### MAIN #################################################
##############################################################################

# l = os.listdir('Train\Mask')  # 1945
# print(len(l))
# l_test = os.listdir('Test\Mask')  # 648
# print(len(l_test))

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.Resize((512, 384)),  # Resize images to match image size
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize for each color channel
    lambda x: x.view(-1),           # Flatten image
    lambda x: x.float()             # Convert to float
])

# Load the training and test datasets
trainset = CustomImageDataset(img_dir=r'Train\Train', labels_csv='metadataTrain.csv', transform=transform, n_elements=600)
testset = CustomImageDataset(img_dir=r'Test\Test', labels_csv='SampleSubmission.csv', transform=transform, n_elements=6333)
image, label, filename = trainset[0]
print(f"train set length: {len(trainset)}")
print(f"test set length: {len(testset)}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

n_hidden_1 = 100 # 1st layer number of neurons
n_hidden_2 = 100 # 2nd layer number of neurons
n_hidden_3 = 50 # 3rd layer number of neurons

n_input = trainset[0][0].shape[0] # input shape (a vectorized 512*384*3 = 589824 dim colored image)
n_output = 8          # ? MNIST total classes (1-8 cases) is it MINST ??????


model_multi_layer = nn.Sequential(
    nn.Linear(n_input, n_hidden_1),
    nn.ReLU(),
    nn.Linear(n_hidden_1, n_hidden_2),
    nn.ReLU(),
    nn.Linear(n_hidden_2, n_hidden_3),
    nn.ReLU(),
    nn.Linear(n_hidden_3, n_output)
)

model_multi_layer = model_multi_layer.to(device)

learning_rate = 0.01
n_epochs = 20
batch_size = 64

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_multi_layer.parameters(), lr = learning_rate)

training_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=False)



model_multi_layer.train()

for epoch in range(0,n_epochs):
  train_loss=0.0
  all_labels = []
  all_predicted = []

  with tqdm(training_loader, unit="batch") as tepoch:
    for data, labels, filename in tepoch:
      tepoch.set_description(f"Epoch {epoch}")

      # Put the data on device
      data = data.to(device)
      labels = labels.to(device)

      # Forward pass: compute predicted outputs by passing inputs to the model
      y_predict = model_multi_layer(data)
    #   print(f"y_predict : {y_predict}")
    #   print(f"labels : {labels}")
      # Calculate the loss
      loss = criterion(y_predict, labels-1)
      
      # Clear the gradients of all optimized variables
      optimizer.zero_grad()
      
      # Perform backward pass: compute gradient of the loss with respect to model parameters
      loss.backward()
      
      # Perform a single optimization step (parameter update)
      optimizer.step()

      # Compute the loss
      train_loss += loss.item()
      # Store labels and class predictions
      all_labels.extend(labels.tolist())
      all_predicted.extend(vector_to_class(y_predict).tolist())

  print('Epoch {}: Train Loss: {:.4f}'.format(epoch, train_loss/len(training_loader.dataset)))
  print('Epoch {}: Train Accuracy: {:.4f}'.format(epoch, prediction_accuracy(np.array(all_predicted),np.array(all_labels))))


input("Press Enter to procede to the test phase...")

model_multi_layer.eval()

all_predicted = []
all_labels = []
all_filenames = []

with tqdm(test_loader, unit="batch") as tepoch:
  for data, labels, filenames in tepoch:
    all_labels.extend(labels.tolist())
    all_filenames.extend(filenames)  # collect the filenames here

    data = data.to(device)
    y_predict = model_multi_layer(data)
    all_predicted.extend(vector_to_class(y_predict).tolist())

test_accuracy = prediction_accuracy(np.array(all_predicted),np.array(all_labels))

print("\nTest Accuracy:", test_accuracy)



# Create a DataFrame
save = True
input(f"Do you wish to save the predictions in a csv file ? Save variable is set to {save}")
if save == True:
  df = pd.DataFrame({
      'ID': all_filenames,
      'CLASS': all_predicted,
      #'LABEL': all_labels
  })

  # Save the DataFrame as a CSV file
  df.to_csv('predictions.csv', index=False)


