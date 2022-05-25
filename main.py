from os.path import exists

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', download=True, train=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', download=True, train=False, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
if exists("model"):
    net.load_state_dict(torch.load("model"))

loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


def train():
    for epoch in range(2):
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            x, y = data

            outputs = net(x)
            l = loss(outputs, y)

            net.zero_grad()
            l.backward()
            optimizer.step()

            running_loss += l.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                print(f'guessed {classes[torch.argmax(outputs[0])]} / actual {classes[y[0]]}')
                running_loss = 0.0
    torch.save(net.state_dict(), 'model')


def show_image(img, guesses):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    for i, guess in enumerate(guesses):
        plt.text(i * 32, -20, guess)
    plt.show()


def test():
    dataiter = iter(testloader)
    x, y = dataiter.next()

    guess_tensor = net(x)
    guesses = []
    for g in guess_tensor:
        guesses.append(classes[torch.argmax(g)])

    show_image(torchvision.utils.make_grid(x), guesses)


IS_TRAINING = False

if IS_TRAINING:
    train()
else:
    test()
