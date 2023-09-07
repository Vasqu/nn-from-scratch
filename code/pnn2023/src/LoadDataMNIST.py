import torch
import torchvision
import torchvision.transforms as transforms
from random import shuffle


# MNIST
train_dataset = torchvision.datasets.MNIST(root="./data", train=True,
                                           transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False,
                                          transform=transforms.ToTensor(), download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=100,
                                           shuffle=True,)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=100,
                                          shuffle=False)


def MNIST_DATA_FNN():
    train = []
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28)
        for j in range(len(images)):
            labels_arr = [0] * 10
            labels_arr[int(float(labels[j].detach().numpy().astype(float)))] = 1

            train.append((images[j].detach().numpy().astype(float), labels_arr))
    # slicing into 10 roughly equally sized chunks
    train = [train[i::10] for i in range(10)]
    dev = train[-1]
    train = train[:-1]
    train = [item for sub_list in train for item in sub_list]

    eval = []
    for i, (images, labels) in enumerate(test_loader):
        images = images.reshape(-1, 28*28)
        for j in range(len(images)):
            labels_arr = [0] * 10
            labels_arr[int(float(labels[j].detach().numpy().astype(float)))] = 1

            eval.append((images[j].detach().numpy().astype(float), labels_arr))

    shuffle(train)
    return train, dev, eval


def MNIST_DATA_CNN():
    train = []
    for i, (images, labels) in enumerate(train_loader):
        for j in range(len(images)):
            labels_arr = [0] * 10
            labels_arr[int(float(labels[j].detach().numpy().astype(float)))] = 1

            train.append((images[j].detach().numpy().astype(float), labels_arr))
    # slicing into 10 roughly equally sized chunks
    train = [train[i::10] for i in range(10)]
    dev = train[-1]
    train = train[:-1]
    train = [item for sub_list in train for item in sub_list]

    eval = []
    for i, (images, labels) in enumerate(test_loader):
        for j in range(len(images)):
            labels_arr = [0] * 10
            labels_arr[int(float(labels[j].detach().numpy().astype(float)))] = 1

            eval.append((images[j].detach().numpy().astype(float), labels_arr))

    shuffle(train)
    return train, dev, eval
