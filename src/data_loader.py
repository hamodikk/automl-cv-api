# src/data_loader.py

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder, DatasetFolder
from PIL import Image

# Data transformation pipeline for the loaded images
def get_transforms(image_size=150, augment=False):
    base_transforms = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]
    
    if augment:
        augmentations = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10)
        ]
        return transforms.Compose(augmentations + base_transforms)

    return transforms.Compose(base_transforms)

# Loader for training images
def get_train_loader(data_dir, image_size=150, batch_size=32, augment=True):
    transform = get_transforms(image_size=image_size, augment=augment)
    dataset = ImageFolder(data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Loader for validation images
def get_test_loader(data_dir, image_size=150, batch_size=32):
    transform = get_transforms(image_size=image_size, augment=False)
    dataset = ImageFolder(data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Custom class to handle prediction images
class UnlabeledImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = [
            os.path.join(root, fname)
            for fname in os.listdir(root)
            if fname.endswith(('.jpg', '.jpeg', '.png'))
        ]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, path
    
# Loader for unlabeled images for prediction
def get_prediction_loader(pred_dir, image_size=150, batch_size=32):
    transform = get_transforms(image_size=image_size, augment=False)
    dataset = UnlabeledImageFolder(pred_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

