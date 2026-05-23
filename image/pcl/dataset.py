from torch.utils.data import Dataset
from torchvision import transforms


class TwoViewTransform:
    """Apply the same stochastic transform twice to produce two augmented views."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class IndexedDataset(Dataset):
    """Wraps a dataset to also return the sample index."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        return data, target, idx

    def __len__(self):
        return len(self.dataset)


# ── Augmentation pipelines ────────────────────────────────────────────────────

def imagenet_train_transform(gaussian_blur: bool = False):
    ops = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
    ]
    if gaussian_blur:
        ops.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], p=0.5))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(ops)


def imagenet_eval_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def cifar10_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])


def cifar10_eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])
