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
from PIL import Image
import os
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

    def __getitem__(self, item):

        image = Image.fromarray(np.float32(np.array(Image.fromarray(plt.imread(self.allImagePaths[item])[:, :, 3]).resize((64, 64))) > 0))
        target = self.allTargets[item]
        image = self.transforms(image)

        return image, target

    def __len__(self):

        return len(self.allImagePaths)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 2, 1)
        self.conv3 = nn.Conv2d(64, 16, 2, 1)
        # self.conv4 = nn.Conv2d(32, 16, 3, 1)
        # self.conv3 = nn.Conv2d(64, 32, 2, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        # x = self.conv4(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        # x = self.conv3(x)
        # x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.dropout2(x)
        x = x.view(x.shape[0],-1)
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
        output = model(data.float()) # Forward pass through the model
        loss = F.cross_entropy(output,target)  # Calculating the loss
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
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Choose the optimizer and the set the learning rate
    testloader = DataLoader(
        customDatasetClass('C:/Users\DELL\Desktop\ML\Machine-Learning\GoogleDataImages_test'),
        batch_size=64,
        num_workers=2,
        shuffle=True
    )
    for epoch in range(1, 15 + 1):
        train(model, use_cuda, trainloader, optimizer, epoch)  # Train the network
        test(model, use_cuda, testloader)  # Test the network

    os.makedirs("data_model", exist_ok=True)
    os.makedirs("save_model", exist_ok=True)
    torch.save(model.state_dict(), "data_model/quickdraw_cnn_epoch.pt")
    dummy_input=torch.zeros(1,1,64,64)
    torch.onnx.export(model,dummy_input,"save_model/model.onnx", verbose = True, input_names = ['data'], output_names = ['output'])
    model.load_state_dict(torch.load('data_model/quickdraw_cnn_epoch.pt'))

if __name__ == "__main__":
    main()
   
