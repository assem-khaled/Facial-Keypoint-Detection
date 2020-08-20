## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # I used the same network structure in the paper "Facial Key Points Detection using Deep Convolutional Neural Network - NaimishNet"
        
        ## Input size ('batch size', 1, 224, 224)
        
        ## output size = (W-F)/S +1 = (224-4)/1 +1 = 221
        # the output Tensor for one image, will have the dimensions: (32, 221, 221)  
        # after one pool layer, this becomes (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.dropout1 = nn.Dropout(0.1)
        
        ## output size = (W-F)/S +1 = (110-3)/1 +1 = 108
        # the output Tensor for one image, will have the dimensions: (64, 108, 108)  
        # after one pool layer, this becomes (64, 54, 54)
        self.conv2 = nn.Conv2d(32, 64, 3)        
        self.dropout2 = nn.Dropout(0.2)
        
        ## output size = (W-F)/S +1 = (54-2)/1 +1 = 53
        # the output Tensor for one image, will have the dimensions: (128, 53, 53)  
        # after one pool layer, this becomes (128, 26, 26)
        self.conv3 = nn.Conv2d(64, 128, 2) 
        self.dropout3 = nn.Dropout(0.3)
        
        ## output size = (W-F)/S +1 = (26-1)/1 +1 = 26
        # the output Tensor for one image, will have the dimensions: (256, 26, 26)  
        # after one pool layer, this becomes (256, 13, 13)
        self.conv4 = nn.Conv2d(128, 256, 1)  
        self.dropout4 = nn.Dropout(0.4)
        
        # Maxpooling Layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(13*13 * 256, 1000) # (43264 => 1000)
        self.dropout5 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(1000, 1000)
        self.dropout6 = nn.Dropout(0.6)
        
        self.fc3 = nn.Linear(1000, 2 * 68) # (1000 => 136)
                        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        # Conv layers
        x = self.dropout1(self.pool(F.relu(self.conv1(x))))
        x = self.dropout2(self.pool(F.relu(self.conv2(x))))
        x = self.dropout3(self.pool(F.relu(self.conv3(x))))
        x = self.dropout4(self.pool(F.relu(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)

        # Fully Connected Layers
        x = self.dropout5(F.relu(self.fc1(x)))
        x = self.dropout6(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
