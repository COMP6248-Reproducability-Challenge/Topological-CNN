from torchvision import transforms, datasets
from torch.utils.data import ConcatDataset, Subset
from torch.utils.data import DataLoader, Dataset
import torch
from os import listdir
from PIL import Image
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split

def load_MINST(download=False):
    trainset = datasets.MNIST(".", train=True, download=download,
                           transform=transforms.Compose([transforms.ToTensor()]))
    testset = datasets.MNIST(".", train=False, download=download,
                          transform=transforms.Compose([transforms.ToTensor()]))
    return ConcatDataset([trainset, testset]) # 70000 samples

def load_SVHN(download=False):
    svhn_transform = transforms.Compose([
        transforms.Resize([28, 28]),  # down-resolve from (32 x 32) to (28 x 28)
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    trainset = datasets.SVHN('SVHN/', split='train', download=download,
                             transform=svhn_transform)
    testset = datasets.SVHN('SVHN/', split='test', download=download,
                            transform=svhn_transform)
    return ConcatDataset([trainset, testset])  # 73257 + 26032 = 99289 samples


def load_CIFAR_cats_dogs(download=False):
    cifar_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ]) # image from [3, 32, 32] to [1, 32, 32]
    trainset = datasets.CIFAR10(root='CIFAR/', train=True, download=True,
                                transform=cifar_transform)
    testset = datasets.CIFAR10(root='CIFAR/', train=False, download=True,
                               transform=cifar_transform)

    # Create a subset of two classes - cats and dogs
    # Use masking to pick samples from the datasets
    train_mask = (torch.tensor(trainset.targets) == trainset.classes.index('cat')) | (
                torch.tensor(trainset.targets) == trainset.classes.index('dog'))
    train_indices = train_mask.nonzero().reshape(-1)
    test_mask = (torch.tensor(testset.targets) == testset.classes.index('cat')) | (
                torch.tensor(testset.targets) == testset.classes.index('dog'))
    test_indices = test_mask.nonzero().reshape(-1)

    train_subset = Subset(trainset, train_indices)
    test_subset = Subset(testset, test_indices)

    return ConcatDataset([train_subset, test_subset])  # 10000 + 2000 = 12000

# Image loader adapted from https://www.kaggle.com/code/adinishad/pytorch-cats-and-dogs-classification/notebook
class ImageLoader(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = self.checkLoadable(dataset) 
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, item):
        image = Image.open(self.dataset[item][0])
        classCategory = self.dataset[item][1]
        if self.transform:
            image = self.transform(image)
        return image, classCategory
        
    
    def checkLoadable(self,dataset):
        datasetLoadable = []
        for index in range(len(dataset)):
            try:
                Image.open(dataset[index][0])
                datasetLoadable.append(dataset[index])
            except (IOError, SyntaxError) as e:
                pass
        return datasetLoadable
    
def load_Kaggle_cats_dogs():
    dataset = ImageFolder("kagglecatsanddogs_3367a/PetImages/")
    train_data, test_data, train_label, test_label = train_test_split(dataset.imgs, dataset.targets, test_size=0.05, random_state=42)
    train_transform = transforms.Compose([transforms.Resize((32, 32)),
                                          transforms.ToTensor(),
                                          transforms.Grayscale()])
    train_dataset = ImageLoader(train_data, train_transform)
    test_dataset = ImageLoader(test_data,train_transform)
    entire_dataset = ConcatDataset([train_dataset, test_dataset])
    return entire_dataset  

