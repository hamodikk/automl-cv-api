# src/train_final.py

import torch
import torch.nn as nn
from torchvision import models
from torch.optim import Adam
from src.data_loader import get_train_loader, get_test_loader
from src.eval_utils import compute_predictions, plot_confusion_matrix, show_misclassified_images
from src import config
import mlflow
import mlflow.pytorch
from tqdm import tqdm

def train_final_model():
    # === CONFIG ===
    epochs = 8
    learning_rate = 2.385058434848458e-05
    optimizer_type = "adam"
    experiment_name = "AutoML_CV_Final"
    
    device = torch.device("cude" if torch.cuda.is_available() else "cpu")
    train_loader = get_train_loader(config.TRAIN_DIR, config.IMAGE_SIZE, config.BATCH_SIZE)
    test_loader = get_test_loader(config.TEST_DIR, config.IMAGE_SIZE, config.BATCH_SIZE)
    
    # === MODEL ===
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(config.CLASS_NAMES))
    model.to(device)
    
    # === OPTIMIZER & LOSS ===
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="best_model_run"):
        mlflow.log_param("optimizer", optimizer_type)
        mlflow.log_param("lr", learning_rate)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("model", "resnet18")
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for images, labels in tqdm(train_loader, leave=False):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            avg_loss = running_loss / len(train_loader)
            y_pred, y_true, _ = compute_predictions(model, test_loader, device)
            val_accuracy = (y_pred == y_true).sum() / len(y_true)
            
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")
            
        # Save the model to MLflow
        mlflow.pytorch.log_model(model, "model")
        mlflow.set_tag("stage", "production")
        
        # Confusion Matrix + Misclassified Images
        print("\n Final evaluation on test set...")
        plot_confusion_matrix(y_true, y_pred, config.CLASS_NAMES)
        show_misclassified_images(_, y_true, y_pred, config.CLASS_NAMES)
        
if __name__ == "__main__":
    train_final_model()