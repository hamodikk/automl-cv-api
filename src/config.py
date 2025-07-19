# src/config.py

# Image dimensions and batch size
IMAGE_SIZE = 150
BATCH_SIZE = 32

# Data directories
TRAIN_DIR = 'data/intel/seg_train/seg_train'
TEST_DIR = 'data/intel/seg_test/seg_test'
PRED_DIR = 'data/intel/seg_pred/seg_pred'

# Class mapping
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']