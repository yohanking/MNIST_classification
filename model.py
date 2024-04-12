
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):

        # write your codes here
        super(LeNet5,self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.dropout = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
    def forward(self, img):

        # write your codes here
        img = F.tanh(self.conv1(img))
        img = F.avg_pool2d(img, 2, 2)
        img = F.tanh(self.conv2(img))
        img = F.avg_pool2d(img, 2, 2)
        img = F.tanh(self.conv3(img))
        img = img.view(-1, 120)
        img = F.tanh(self.fc1(img))
        self.dropout = nn.Dropout(0.5)
        img = self.fc2(img)
        output = F.softmax(img, dim=1)
        return output


class CustomMLP(nn.Module):
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
        - with LeNet-5
    """
    def __init__(self):

        # write your codes here
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(1024, 60)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(60, 10)    
    def forward(self, img):

        # write your codes here
        output = F.relu(self.fc1(img))
        self.dropout = nn.Dropout(0.5)
        output = self.fc2(output)
        output = output.squeeze(dim=1)
        return output
