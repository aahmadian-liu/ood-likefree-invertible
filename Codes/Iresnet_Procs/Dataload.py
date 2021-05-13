import torch
import torchvision
import torchvision.transforms as transforms
import numpy as  np


dens_est_chain = [
        lambda x: (255. * x) + torch.zeros_like(x).uniform_(0., 1.),
        lambda x: x / 256.,
        lambda x: x - 0.5
    ]
transform_test = transforms.Compose([transforms.ToTensor()] + dens_est_chain)

cifar10_train=  torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_test)
cifar10_test= torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)

cifar100_train=  torchvision.datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=transform_test)
cifar100_test= torchvision.datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform_test)

svhn_train = torchvision.datasets.SVHN(root='./data/svhn', split='train', download=True, transform=transform_test)
svhn_test = torchvision.datasets.SVHN(root='./data/svhn', split='test', download=True, transform=transform_test)

transform_test_mnist= transforms.Compose([transforms.Pad(2, 0), transforms.ToTensor(), lambda x: x.repeat((3, 1, 1))]+dens_est_chain)

mnist_test= torchvision.datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform_test_mnist)
fashion_test= torchvision.datasets.FashionMNIST(root='./data/fashion', train=False, download=True, transform=transform_test_mnist)

mnist_train= torchvision.datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform_test_mnist)
fashion_train= torchvision.datasets.FashionMNIST(root='./data/fashion', train=True, download=True, transform=transform_test_mnist)

batchsize=1

def Loader(dataset):
    return torch.utils.data.DataLoader(dataset, batch_size=batchsize,shuffle=False, num_workers=1)

def Blankloader():
    while True:
        r=torch.rand(1)
        im=(torch.ones([batchsize,3,32,32])*r)-0.5
        yield (im,torch.zeros(batchsize,1))

def Whitenoiseloader():
    while True:
        r=torch.randn([batchsize,3,32,32])-0.5
        yield (r,torch.zeros(batchsize,1))

def Uninoiseloader():
    while True:
        r=torch.rand([batchsize,3,32,32])-0.5
        yield (r,torch.zeros(batchsize,1))

def Ranblockloader(dataset):

    loader=torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    for img,_ in loader:

        per=np.array(range(0,16))
        np.random.shuffle(per)

        imgp=torch.zeros_like(img)

        for i in range(0,16):
            t=per[i]

            y=(i//4)*8
            x=(i%4)*8
            yt = (t//4)*8
            xt = (t % 4)*8

            imgp[:,:,y:y+8,x:x+8]=img[:,:,yt:yt+8,xt:xt+8]


        yield (imgp,torch.tensor(0))


