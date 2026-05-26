"""
Preprocessing utilities for LSTM sign language recognition
"""
import numpy as np
import pickle as pkl
from typing import List, Tuple


def standardize_sequence(sequence: np.ndarray, target_length: int = 30) -> np.ndarray:
    """
    Standardize sequence length by padding or truncating
    
    Args:
        sequence: Array of shape (frames, features)
        target_length: Target number of frames
    
    Returns:
        Standardized array of shape (target_length, features)
    """
    current_length = len(sequence)
    
    if current_length == target_length:
        return sequence
    
    elif current_length > target_length:
        # Truncate: take evenly spaced frames
        indices = np.linspace(0, current_length - 1, target_length, dtype=int)
        return sequence[indices]
    
    else:
        # Pad with zeros
        features = sequence.shape[1]
        padded = np.zeros((target_length, features))
        padded[:current_length] = sequence
        return padded


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize landmarks to be translation and scale invariant
    
    Args:
        landmarks: Array of shape (frames, 63) or (frames, 126) for both hands
    
    Returns:
        Normalized landmarks
    """
    # Reshape to (frames, num_points, 3)
    frames, features = landmarks.shape
    num_points = features // 3
    reshaped = landmarks.reshape(frames, num_points, 3)
    
    normalized_frames = []
    for frame in reshaped:
        # Skip if all zeros (no hand detected)
        if np.sum(frame) == 0:
            normalized_frames.append(frame)
            continue
        
        # Center around wrist (first landmark)
        centered = frame - frame[0]
        
        # Scale by max distance from wrist
        max_dist = np.max(np.linalg.norm(centered, axis=1))
        if max_dist > 0:
            scaled = centered / max_dist
        else:
            scaled = centered
        
        normalized_frames.append(scaled)
    
    # Reshape back to (frames, features)
    normalized = np.array(normalized_frames).reshape(frames, features)
    return normalized


def load_pickle_file(filepath: str) -> np.ndarray:
    """Load landmarks from pickle file"""
    with open(filepath, 'rb') as f:
        data = pkl.load(f)
    return np.array(data)


def combine_hands(left_hand: np.ndarray, right_hand: np.ndarray) -> np.ndarray:
    """
    Combine left and right hand landmarks
    
    Args:
        left_hand: Array of shape (frames, 63)
        right_hand: Array of shape (frames, 63)
    
    Returns:
        Combined array of shape (frames, 126)
    """
    return np.concatenate([left_hand, right_hand], axis=1)
