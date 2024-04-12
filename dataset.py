
# import some packages you need here
import os
from PIL import Image
import glob
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class MNIST(Dataset):

    def __init__(self, data_dir, model=None):

        # write your codes here
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
            ])
        self.images = glob.glob(self.data_dir + '/*.png')
        self.labels = [int(file_name.split('_')[-1][0]) for file_name in self.images]
        self.model = model
    def __len__(self):
        
        # write your codes here
        return len(self.images)
    def __getitem__(self, idx):

        # write your codes here
        img = Image.open(self.images[idx])
        img = self.transform(img)
        img = np.array(img)
        if self.model == 'LeNet5':
            img = img
        else:
            img = img.reshape(1, -1)
            
        label = self.labels[idx]

        return img, label

if __name__ == '__main__':

    # write test codes to verify your implementations
    test_data_dir = 'your_dataset_dir'
    train_data_dir = 'your_dataset_dir'
    test_data_count = len(os.listdir(test_data_dir))
    train_data_count = len(os.listdir(train_data_dir))
    print(f"Number of data samples in the test directory: {test_data_count}")
    print(f"Number of data samples in the train directory: {train_data_count}")    
