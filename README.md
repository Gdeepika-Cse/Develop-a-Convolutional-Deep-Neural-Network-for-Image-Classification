# NAME: DEEPIKA G
# REG.NO: 212224040060
# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Include the Problem Statement and Dataset.

## Neural Network Model

<img width="1066" height="499" alt="586405134-7a35a71c-35e1-4d05-814d-7ea6b0498eea" src="https://github.com/user-attachments/assets/aa7ec169-4cb5-4367-825d-07f1eed06abc" />

## DESIGN STEPS
## STEP 1:
Import the required libraries (torch, torchvision, torch.nn, torch.optim) and load the image dataset with necessary preprocessing like normalization and transformation.

## STEP 2:
Split the dataset into training and testing sets and create DataLoader objects to feed images in batches to the CNN model.

## STEP 3:
Define the CNN architecture using convolutional layers, ReLU activation, max pooling layers, and fully connected layers as implemented in the CNNClassifier class.

## STEP 4:
Initialize the model, define the loss function (CrossEntropyLoss), and choose the optimizer (Adam) for training the network.

## STEP 5:
Train the model using the training dataset by performing forward pass, computing loss, backpropagation, and updating weights for multiple epochs.

## STEP 6:
Evaluate the trained model on test images and verify the classification accuracy for new unseen images.

## PROGRAM

### Name:DEEPIKA G

### Register Number:212224040060

```
python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # write your code here
        self.conv1=nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2=nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3=nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1=nn.Linear(128*3*3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)
    def forward(self,x):
      x=self.pool(torch.relu(self.conv1(x)))
      x=self.pool(torch.relu(self.conv2(x)))
      x=self.pool(torch.relu(self.conv3(x)))
      x=x.view(x.size(0),-1)
      x=torch.relu(self.fc1(x))
      x=torch.relu(self.fc2(x))
      x=self.fc3(x)
      return x

model =CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(), lr=0.001)


## Step 3: Train the Model
def train_model(model, train_loader, num_epochs=3):

    # write your code here
    for epoch in range(num_epochs):
      model.train()
      running_loss = 0.0
      for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss=criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

      print('Name:DEEPIKA G')
      print('Register Number:212224040060')
      print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")
    
```

### OUTPUT

## Training Loss per Epoch

<img width="500" height="307" alt="image" src="https://github.com/user-attachments/assets/a4d32cf5-1cba-46ce-82c8-fb8771d2f6e6" />

## Confusion Matrix

<img width="851" height="793" alt="image" src="https://github.com/user-attachments/assets/a5a3f101-6ae2-46b9-a852-b1c6cd4daeaa" />

## Classification Report

<img width="629" height="412" alt="image" src="https://github.com/user-attachments/assets/40c86c2c-fb4e-4d36-ab69-8c5a97570984" />

### New Sample Data Prediction

<img width="572" height="635" alt="image" src="https://github.com/user-attachments/assets/c2272845-62ec-4e9c-a521-1c93787e8b92" />

## RESULT
Thus , a convolutional deep neural network (CNN) for image classification and to verify the response for new images is successfully developed.
