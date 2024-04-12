# import some packages you need here
import dataset
from model import LeNet5, CustomMLP
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')


def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """

    # write your codes here
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in trn_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    trn_loss = running_loss / len(trn_loader)
    
    acc = 100. * correct / total
    
    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """

    # write your codes here
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tst_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    tst_loss = running_loss / len(tst_loader)
    acc = 100. * correct / total
    return tst_loss, acc

def plot_loss_over_epochs(epochs,status,model, value, l_or_a):
    
    plt.plot(epochs, value)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(model +'_'+ status +'_' + l_or_a)
    plt.savefig("/dshome/ddualab/yohan/deeplearning_HW/img/plot_{}_{}_{}.png".format(model, status, l_or_a))
    plt.clf()
    
    
def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """

    # write your codes here
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dir = '/dshome/ddualab/yohan/deeplearning_HW/data/train/train'
    test_dir = '/dshome/ddualab/yohan/deeplearning_HW/data/test/test'
    model_type = 'LeNet5'  # 'LeNet5' or 'CustomMLP'
    
    train_dataset = dataset.MNIST(data_dir=train_dir, model=model_type)
    test_dataset = dataset.MNIST(data_dir=test_dir, model=model_type)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4)
    
    if model_type == 'LeNet5':
        model = LeNet5().to(device)
    elif model_type == 'CustomMLP':
        model = CustomMLP().to(device)
        
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum = 0.9)
    real_epoch = 20
    
    epochs = [i+1 for i in range(real_epoch)]
    train_loss_ = []
    test_loss_ = []
    train_acc_ = []
    test_acc_ = []
    
    for epoch in range(real_epoch):
        train_loss, train_acc = train(model, train_loader, device, criterion, optimizer)
        train_loss_.append(train_loss)
        train_acc_.append(train_acc)
        print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        test_loss, test_acc = test(model, test_loader, device, criterion)
        test_loss_.append(test_loss)
        test_acc_.append(test_acc)
        print(f"Epoch {epoch + 1}: Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    #save loss & accuracy    
    plot_loss_over_epochs(epochs,'train',model_type,train_loss_,'loss')
    plot_loss_over_epochs(epochs,'test',model_type,test_loss_, 'loss')
    plot_loss_over_epochs(epochs,'train',model_type,train_acc_, 'acc')
    plot_loss_over_epochs(epochs,'test',model_type,test_acc_, 'acc')



if __name__ == '__main__':
    main()
