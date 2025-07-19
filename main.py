# main.py

from src import config
from src.data_loader import get_train_loader, get_test_loader, get_prediction_loader

train_loader = get_train_loader(config.TRAIN_DIR)
test_loader = get_test_loader(config.TEST_DIR)
pred_loader = get_prediction_loader(config.PRED_DIR)

# Inspect one batch from each loader
images, labels = next(iter(train_loader))
print("Train batch:", images.shape, labels.shape)

test_images, test_labels = next(iter(test_loader))
print("Test batch:", test_images.shape, test_labels.shape)

pred_images, pred_paths = next(iter(pred_loader))
print("Prediction batch:", pred_images.shape)
print("Sample file paths:", pred_paths[:3])