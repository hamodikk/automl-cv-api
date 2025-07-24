# src/eval_utils.py

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import os

# Utility functions for model evaluation
def compute_predictions(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_images = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_images.extend(images.cpu())
            
    return np.array(all_preds), np.array(all_labels), all_images

# Function for plotting confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8,6))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
        
# Function to show misclassified images
def show_misclassified_images(images, y_true, y_pred, class_names, max_samples=6):
    wrong_indices = np.where(y_true != y_pred)[0]
    if len(wrong_indices) == 0:
        print("No misclassified images found.")
        return
    
    # Pick misclassified images at random
    selected = np.random.choice(wrong_indices, min(max_samples, len(wrong_indices)), replace=False)
    
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(selected):
        image = images[idx].permute(1, 2, 0).numpy()
        image = (image * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406]) # Unnormalize using ImageNet std/mean
        image = np.clip(image, 0, 1)
        
        plt.subplot(2, 3, i + 1)
        plt.imshow(image)
        plt.title(f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}")
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()