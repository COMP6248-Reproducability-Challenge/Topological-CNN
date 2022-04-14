from torchvision import transforms, datasets
from torch.utils.data import ConcatDataset

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