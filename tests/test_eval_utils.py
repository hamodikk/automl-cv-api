import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from src.eval_utils import compute_predictions

def test_compute_predictions_runs():
    # Dummy model
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    
    #Dummy data
    images = torch.randn(8, 3, 150, 150) # 8 images, 3 channels, 150x150 pixels
    labels = torch.randint(0, 2, (8,))
    
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=4)
    
    preds, true, imgs = compute_predictions(model, dataloader, device="cpu")
    
    assert len(preds) == len(true) == len(imgs)