# src/tune.py

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from src.data_loader import get_train_loader, get_test_loader
from src.eval_utils import compute_predictions
from src import config
import mlflow
import mlflow.pytorch
from tqdm import tqdm

# Function to train and evaluate the model
def train_and_evaluate(params, trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = get_train_loader(config.TRAIN_DIR, config.IMAGE_SIZE, config.BATCH_SIZE)
    test_loader = get_test_loader(config.TEST_DIR, config.IMAGE_SIZE, config.BATCH_SIZE)
    
    # Model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(config.CLASS_NAMES))
    model = model.to(device)
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if params["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=params["lr"], momentum=0.9)
        
    # Training loop
    for epoch in range(params["epochs"]):
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
            
    # Evaluation
    y_pred, y_true, _ = compute_predictions(model, test_loader, device)
    accuracy = (y_pred == y_true).sum() / len(y_true)
    
    # Report score to Optuna
    trial.report(accuracy, step=epoch)
    return accuracy

# Optuna objective function
def objective(trial):
    # Define seach space
    params = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd"]),
        "epochs": trial.suggest_int("epochs", 5, 15)
    }
    
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        accuracy = train_and_evaluate(params, trial)
        mlflow.log_metric("val_accuracy", accuracy)
    return accuracy

# Run the Optuna study
def run_study():
    mlflow.set_experiment("AutoML-CV-Optuna")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    
    print("\nBest hyperparameters:")
    print(study.best_trial.params)
    print(f"Best accuracy: {study.best_value:.4f}")
    
if __name__ == "__main__":
    run_study()