import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from PIL import Image,ImageOps
import os
# from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class customDatasetClass(Dataset):

    def __init__(self, path):

        self.path = path
        self.allImagePaths = []
        self.allTargets = []
        self.targetToClass = sorted(os.listdir(self.path))

        for targetNo, targetI in enumerate(self.targetToClass):
            for imageI in sorted(os.listdir(self.path + '/' + targetI + '/image')):
                self.allImagePaths.append(self.path + '/' + targetI + '/image/' + imageI)
                self.allTargets.append(targetNo)

        self.transforms = torchvision.transforms.Compose([
            # torchvision.transforms.RandomCrop((256, 256)),
            torchvision.transforms.ToTensor()
        ])
        # allImagePaths=allImagePaths.reshape(allImagePaths.shape[0],3,32,32)

    def __getitem__(self, item):

        image = Image.open(self.allImagePaths[item]).convert('RGB')
        image1 = ImageOps.grayscale(image) 
        image1=image1.resize((32,32))
        # image1.show()
        # exit(0)
        target = self.allTargets[item]
        image1 = self.transforms(image1)

        return image1, target

    def __len__(self):

        return len(self.allImagePaths)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 2, 1)
        # self.conv3 = nn.Conv2d(64, 128, 2, 1)
        # self.conv4 = nn.Conv2d(96, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 32, 2, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(6272, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        # print(x.shape)
        # exit(0)
        x = self.conv1(x)
        x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        # print(x.shape)
        # exit(0)
        x = self.conv2(x)
        x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        # x = self.conv3(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        # x = self.conv4(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.dropout1(x)
        x = x.view(x.shape[0],-1)
        # print(x.shape)
        # exit(0)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def train(model, use_cuda, train_loader, optimizer, epoch):

    model.train()  # Tell the model to prepare for training

    for batch_idx, (data, target) in enumerate(train_loader):  # Get the batch

        if use_cuda:
            data, target = data.cuda(), target.cuda()  # Sending the data to the GPU

        optimizer.zero_grad()  # Setting the cumulative gradients to 0
        output = model(data.float())
        # print(output.shape,target.shape)# Forward pass through the model
        loss = F.cross_entropy(output,target.long())  # Calculating the loss
        loss.backward()  # Calculating the gradients of the model. Note that the model has not yet been updated.
        optimizer.step()  # Updating the model parameters. Note that this does not remove the stored gradients!

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, use_cuda, test_loader):

    model.eval()  # Tell the model to prepare for testing or evaluation

    test_loss = 0
    correct = 0

    with torch.no_grad():  # Tell the model that gradients need not be calculated
        for data, target in test_loader:  # Get the batch

            if use_cuda:
                data, target = data.cuda(), target.cuda()  # Sending the data to the GPU

            # argmax([0.1, 0.2, 0.9, 0.4]) => 2
            # output - shape = [1000, 10], argmax(dim=1) => [1000]
            output = model(data)  # Forward pass
            test_loss += F.cross_entropy(output,target.long(),reduction='sum')  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the maximum output
            correct += pred.eq(target.view_as(pred)).sum().item()  # Get total number of correct samples

    test_loss /= len(test_loader.dataset)  # Accuracy = Total Correct / Total Samples

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def seed(seed_value):

    # This removes randomness, makes everything deterministic

    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def main():

    use_cuda = False  # Set it to False if you are using a CPU
    # Colab And Kaggle

    seed(0)

    transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    trainloader = DataLoader(
        customDatasetClass('C:/Users/DELL/Desktop/ML/Machine-Learning/GoogleDataImages_train'),
        batch_size=64,
        num_workers=2,
        shuffle=True
    )

    model=Net()
    optimizer = optim.SGD(model.parameters(), lr=0.001)  # Choose the optimizer and the set the learning rate
    testloader = DataLoader(
        customDatasetClass('C:/Users\DELL\Desktop\ML\Machine-Learning\GoogleDataImages_test'),
        batch_size=64,
        num_workers=2,
        shuffle=True
    )
    for epoch in range(1, 10 + 1):
        train(model, use_cuda, trainloader, optimizer, epoch)  # Train the network
        test(model, use_cuda, testloader)  # Test the network


    torch.save(model.state_dict(), "quickdraw_model.pt")

    model.load_state_dict(torch.load('quickdraw_model.pt'))

if __name__ == "__main__":
    main()
   
