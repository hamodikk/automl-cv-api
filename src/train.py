# src/train.py

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torchvision import models
from torch.optim import Adam
from src.data_loader import get_train_loader, get_test_loader
from src.eval_utils import compute_predictions, plot_confusion_matrix, show_misclassified_images
from src import config
from tqdm import tqdm

# Evaluation helper function
def evaluate(model, dataloader, device):
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return correct / total
    
def train():
    # Hyperparameters
    epochs = 5
    learning_rate = 0.001
    
    # MLflow setup
    mlflow.set_experiment("AutoML-CV-Baseline")
    with mlflow.start_run():
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("lr", learning_rate)
        mlflow.log_param("model", "resnet18")
        
        # Data
        train_loader = get_train_loader(config.TRAIN_DIR, config.IMAGE_SIZE, config.BATCH_SIZE)
        test_loader = get_test_loader(config.TEST_DIR, config.IMAGE_SIZE, config.BATCH_SIZE)
        
        # Device configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model
        model = models.resnet18(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(config.CLASS_NAMES))
        model = model.to(device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            avg_loss = running_loss / len(train_loader)
            mlflow.log_metric("loss", avg_loss, step=epoch)
            
            accuracy = evaluate(model, test_loader, device)
            mlflow.log_metric("accuracy", accuracy, step=epoch)
            
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}, Val Accuracy: {accuracy:.4f}")
            
        # Save model to MLflow
        mlflow.pytorch.log_model(model, "model")
        
        print("\n Running final evaluation...")

        y_pred, y_true, images = compute_predictions(model, test_loader, device)
        plot_confusion_matrix(y_true, y_pred, config.CLASS_NAMES)
        show_misclassified_images(images, y_true, y_pred, config.CLASS_NAMES)
        
if __name__ == "__main__":
    train()
    print("Training complete. Model saved to MLflow.")