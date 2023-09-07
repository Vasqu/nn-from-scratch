import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


# force torch to use cpu
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device("cpu")


# MNIST
train_dataset = torchvision.datasets.MNIST(root="./data", train=True,
                                           transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False,
                                          transform=transforms.ToTensor(), download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=1,
                                           shuffle=True,)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=1,
                                          shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=9)
        self.pool = nn.AvgPool2d(kernel_size=4, stride=4)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x


model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


if __name__ == '__main__':
    # training loop
    n_total_steps = len(train_dataset)
    for epoch in range(1):
        for i, (image, label) in enumerate(train_loader):

            # forward
            outputs = model(image)
            loss = criterion(outputs, label)

            # backwards
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if(i%100) == 0:
                print(f"epoch {epoch+1}/{1}, step {i+1}/{n_total_steps} loss = {loss.item():.4f}")


    # test
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for image, label in test_loader:
            output = model(image)
            # value, index
            _, prediction = torch.max(output, 1)
            n_correct += (prediction == label)
            n_samples += 1

        acc = n_correct / n_samples
        print(f"accuracy = {acc}")