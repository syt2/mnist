import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def mnistDataLoader(data_dir='dataset',is_train=True,batch_size=64,shuffle=True,workers=2):
    if is_train:
        trans = [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))]
        trans = transforms.Compose(trans)
        train_set = datasets.MNIST(data_dir, train=True, transform=trans, download=True)
        loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=shuffle,num_workers=workers, pin_memory=True)
    else:
        trans = [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))]
        trans = transforms.Compose(trans)
        test_set = datasets.MNIST(data_dir, train=False, transform=trans)
        loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=shuffle,num_workers=workers, pin_memory=True)
    return loader