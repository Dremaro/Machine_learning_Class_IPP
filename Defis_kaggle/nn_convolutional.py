import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Normalize
import torch.utils.data

#region ########################## FUNCTIONS and CLASSES ##############################
#######################################################################################

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, labels_csv, transform=None):
        self.img_dir = img_dir
        self.labels_df = pd.read_csv(labels_csv)
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labels_df.iloc[idx, 0] + ".jpg")
        image = Image.open(img_path).convert('RGB')
        label = self.labels_df.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label
#endregion

############################## VARIABLES and DATA ##############################
################################################################################



# Define the transform
transform = transforms.Compose([
    transforms.Resize((384, 512)),  # Resize images to match the CIFAR-10 dataset
    ToTensor(),  # Convert PIL images to PyTorch tensors
    #Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images to the range [-1, 1]
])

# Load training and test data
trainset_full = CustomImageDataset('Train/Train', 'metadataTrain.csv', transform=transform)
testset_full = CustomImageDataset('Test/Test', 'SampleSubmission.csv', transform=transform)

# Create subsets
trainset = torch.utils.data.Subset(trainset_full, range(500))  # First 100 examples
testset = torch.utils.data.Subset(testset_full, range(1000))  # First 50 examples

# Extract the actual data and labels
X_train = torch.stack([img for img, _ in tqdm(trainset)])
Y_train = torch.tensor([label for _, label in tqdm(trainset)])
X_test = torch.stack([img for img, _ in tqdm(testset)])
Y_test = torch.tensor([label for _, label in tqdm(testset)])

nb_channels = X_train.shape[1]  # 3 for RGB images



### Building the model ###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

learning_rate = 0.01
n_epochs = 40
batch_size = 256
nb_classes = 10

nb_filters = 32         # number of convolutional filters to use
kernel_size = (3, 3)    # convolution kernel size
pool_size = (2, 2)      # size of pooling area for max pooling

# convolutionnal layer : output_size = (input_size - kernel_size) / stride + 1
# max_pooling layer : output_size = input_size / pool_size

# --- Size of the successive layers
n_h_0 = nb_channels
n_h_1 = nb_filters
n_h_2 = nb_filters
n_h_3 = nb_filters

mon_model = torch.nn.Sequential(
    torch.nn.Conv2d(n_h_0, n_h_1, kernel_size=kernel_size, stride=1, padding=1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(n_h_1, n_h_2, kernel_size=kernel_size, stride=1, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=pool_size, stride=2),
    torch.nn.Conv2d(n_h_2, n_h_3, kernel_size=kernel_size, stride=1, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=pool_size, stride=2),
    torch.nn.Conv2d(n_h_3, n_h_3, kernel_size=kernel_size, stride=1, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=pool_size, stride=2),
    torch.nn.Flatten(),
    torch.nn.Linear(64*48*n_h_3, nb_classes)
)
#mon_model = mon_model.to(device)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mon_model.parameters(), lr=learning_rate)

### Training the model ###
input('train the model : press any key to continue')
train_losses=[]
valid_losses=[]

for epoch in tqdm(range(0,n_epochs)):
  train_loss=0.0
  all_labels = []
  all_predicted = []

  for batch_idx, (imgs, labels) in enumerate(train_loader):
    #imgs = imgs.to(device)
    #labels = labels.to(device)
    
    # pass the samples through the network
    predict = mon_model(imgs)
    # apply loss function
    loss = criterion(predict, labels)
    # set the gradients back to 0
    optimizer.zero_grad()
    # backpropagation
    loss.backward()
    # parameter update
    optimizer.step()
    # compute the train loss
    train_loss += loss.item()
    # store labels and class predictions
    all_labels.extend(labels.tolist())
    all_predicted.extend(predict.argmax(dim=1).tolist())

  print('Epoch:{} Train Loss:{:.4f}'.format(epoch,train_loss/len(train_loader.dataset)))

  # calculate accuracy
  correct_predictions = (np.array(all_predicted) == np.array(all_labels))
  accuracy = np.mean(correct_predictions)
  print('Accuracy:{:.4f}'.format(accuracy))



input('test the model : press any key to continue')


from torch.utils.data import DataLoader, TensorDataset

# Create DataLoader for training and test data
batch_size = 64  # adjust this value according to your system's memory
train_data = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_data, batch_size=batch_size)
test_data = TensorDataset(X_test, Y_test)
test_loader = DataLoader(test_data, batch_size=batch_size)

# Calculate accuracy on the test set
correct = 0
total = 0
all_predicted = []
for data, labels in test_loader:
    data, labels = data.to(device), labels.to(device)
    outputs = mon_model(data)
    _, predicted = torch.max(outputs.data, 1)
    all_predicted.extend(predicted.cpu().tolist())  # move predictions back to CPU for further processing
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
test_accuracy = correct / total

print("Test Accuracy:", test_accuracy)

# Save test predictions to a CSV file
df = pd.DataFrame({
    'ID': range(len(all_predicted)),  # or replace with actual IDs if you have them
    'CLASS': all_predicted,
})
df.to_csv('predictions.csv', index=False)













































# Calculate accuracy on the training set and the test set
# X_train = X_train.to(device)
# Y_train = Y_train.to(device)
# X_test = X_test.to(device)
# Y_test = Y_test.to(device)
# mon_model = mon_model.to("cpu")
# predict_train = mon_model(X_train).argmax(dim=1)
# train_accuracy = (predict_train == Y_train).float().mean().item()

# predict_test = mon_model(X_test).argmax(dim=1)
# test_accuracy = (predict_test == Y_test).float().mean().item()

# print("Train Accuracy:", train_accuracy)
# print("Test Accuracy:", test_accuracy)


# # Save test predictions to a CSV file
# predicted_list = predict_test.tolist()
# df = pd.DataFrame({
#     'ID': range(len(predicted_list)),  # or replace with actual IDs if you have them
#     'CLASS': predicted_list,
# })
# df.to_csv('predictions.csv', index=False)






exit()



input ('visualise results : press any key to continue')

plt.figure(figsize=(10, 6))
for num in range(0,32):
    plt.subplot(8, 4, num+1)
    # --- START CODE HERE
    filter = mon_model[0].weight[num, 0].detach().numpy()
    plt.imshow(filter, cmap='gray')
    # --- END CODE HERE
plt.show()







