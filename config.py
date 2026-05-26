"""
Configuration file for LSTM-based Sign Language Recognition
Modify these paths to match your setup
"""

# Data paths
VIDEOS_PATH = r"D:\Study\Sign-Language-Recognition--MediaPipe-DTW-master\our_dataset"  # Path to reference videos
DATASET_PATH = r"D:\Study\Sign-Language-Recognition--MediaPipe-DTW-master\data\dataset"  # Path to extracted landmarks

# MediaPipe settings
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# LSTM Model settings
SEQUENCE_LENGTH = 30  # Number of frames per sequence
FEATURE_DIM = 126  # 63 per hand (21 landmarks × 3) × 2 hands
BATCH_SIZE = 16  # Training batch size
EPOCHS = 50  # Training epochs

# Prediction settings
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for prediction (0.0 to 1.0)

# Camera settings
CAMERA_INDEX = 0  # Camera device index (0, 1, 2, etc.) - Try 0 first, then 1 or 2
CAMERA_BACKEND = "CAP_DSHOW"  # Windows: CAP_DSHOW, Linux: CAP_V4L2, Mac: CAP_AVFOUNDATION

# Display settings
DISPLAY_HEIGHT = 600
