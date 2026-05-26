"""
Predictor Module
Handles video processing, landmark extraction, and sign prediction
Uses your existing preprocessing and MediaPipe logic
"""
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import os
from typing import Tuple, List, Dict, Any
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

# Import your existing utilities (no modifications)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.mediapipe_utils import mediapipe_detection
from utils.landmark_utils import extract_landmarks
from utils.preprocessing import normalize_landmarks, standardize_sequence


class SignLanguagePredictor:
    """
    Handles sign language prediction from video files
    Uses MediaPipe for landmark extraction and GRU model for classification
    """
    
    def __init__(
        self,
        model: keras.Model,
        label_encoder: LabelEncoder,
        sign_classes: List[str],
        sequence_length: int = 30,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize the predictor
        
        Args:
            model: Trained Keras model
            label_encoder: Sklearn label encoder for class decoding
            sign_classes: List of sign class names
            sequence_length: Number of frames to use for prediction
            confidence_threshold: Minimum confidence to accept prediction
        """
        self.model = model
        self.label_encoder = label_encoder
        self.sign_classes = sign_classes
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        
        # MediaPipe configuration (from your config.py)
        self.min_detection_confidence = 0.5
        self.min_tracking_confidence = 0.5
        
    def extract_landmarks_from_video(self, video_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract hand landmarks from video file using MediaPipe
        This uses your existing landmark extraction logic
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (landmarks_array, metadata_dict)
            landmarks_array: Shape (num_frames, 126) - 63 per hand
            metadata_dict: Information about extraction quality
            
        Raises:
            ValueError: If video cannot be opened or no hands detected
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Storage for landmarks
        left_hand_landmarks = []
        right_hand_landmarks = []
        
        # Quality tracking
        total_frames = 0
        frames_with_hands = 0
        frames_with_left = 0
        frames_with_right = 0
        
        try:
            # Initialize MediaPipe Holistic
            with mp.solutions.holistic.Holistic(
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            ) as holistic:
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    total_frames += 1
                    
                    # Make detections using your existing function
                    image, results = mediapipe_detection(frame, holistic)
                    
                    # Extract landmarks using your existing function
                    _, left_hand, right_hand = extract_landmarks(results)
                    
                    # Track detection quality
                    if results.left_hand_landmarks or results.right_hand_landmarks:
                        frames_with_hands += 1
                    if results.left_hand_landmarks:
                        frames_with_left += 1
                    if results.right_hand_landmarks:
                        frames_with_right += 1
                    
                    # Store landmarks
                    left_hand_landmarks.append(left_hand)
                    right_hand_landmarks.append(right_hand)
                    
        finally:
            cap.release()
        
        # Validate extraction
        if total_frames == 0:
            raise ValueError("No frames extracted from video")
        
        if frames_with_hands == 0:
            raise ValueError("No hands detected in any frame of the video")
        
        # Combine left and right hand landmarks
        # Each frame: 63 (left) + 63 (right) = 126 features
        combined_landmarks = []
        for left, right in zip(left_hand_landmarks, right_hand_landmarks):
            combined = np.array(left + right)  # Concatenate
            combined_landmarks.append(combined)
        
        landmarks_array = np.array(combined_landmarks)
        
        # Metadata about extraction
        metadata = {
            'total_frames': total_frames,
            'frames_with_hands': frames_with_hands,
            'frames_with_left': frames_with_left,
            'frames_with_right': frames_with_right,
            'hand_detection_rate': round(frames_with_hands / total_frames * 100, 2),
            'extracted_shape': landmarks_array.shape
        }
        
        return landmarks_array, metadata
    
    def preprocess_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Preprocess landmarks for model input
        Uses your existing preprocessing functions
        
        Args:
            landmarks: Raw landmarks array (num_frames, 126)
            
        Returns:
            Preprocessed array ready for model (1, sequence_length, 126)
        """
        # Standardize sequence length using your existing function
        standardized = standardize_sequence(landmarks, target_length=self.sequence_length)
        
        # Normalize landmarks using your existing function
        normalized = normalize_landmarks(standardized)
        
        # Add batch dimension for model input
        preprocessed = np.expand_dims(normalized, axis=0)
        
        return preprocessed
    
    def predict(self, video_path: str) -> Dict[str, Any]:
        """
        Predict sign language from video file
        Complete pipeline: extract landmarks -> preprocess -> predict
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with prediction results:
            {
                'predicted_sign': str,
                'confidence': float,
                'all_predictions': dict,
                'metadata': dict
            }
            
        Raises:
            ValueError: If video processing fails
        """
        # Step 1: Extract landmarks from video
        try:
            landmarks, extraction_metadata = self.extract_landmarks_from_video(video_path)
        except Exception as e:
            raise ValueError(f"Landmark extraction failed: {str(e)}")
        
        # Step 2: Preprocess landmarks
        try:
            preprocessed = self.preprocess_landmarks(landmarks)
        except Exception as e:
            raise ValueError(f"Preprocessing failed: {str(e)}")
        
        # Step 3: Make prediction
        try:
            predictions = self.model.predict(preprocessed, verbose=0)[0]
        except Exception as e:
            raise ValueError(f"Model prediction failed: {str(e)}")
        
        # Step 4: Decode prediction
        predicted_class_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_class_idx])
        
        # Decode label using your label encoder
        predicted_sign = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        top_3_predictions = {}
        for idx in top_3_indices:
            sign_name = self.label_encoder.inverse_transform([idx])[0]
            top_3_predictions[sign_name] = float(predictions[idx])
        
        # Build result
        result = {
            'predicted_sign': predicted_sign,
            'confidence': confidence,
            'confidence_percentage': round(confidence * 100, 2),
            'meets_threshold': confidence >= self.confidence_threshold,
            'threshold': self.confidence_threshold,
            'top_3_predictions': top_3_predictions,
            'extraction_metadata': extraction_metadata
        }
        
        return result
    
    def predict_from_bytes(self, video_bytes: bytes) -> Dict[str, Any]:
        """
        Predict sign language from video bytes (for API upload)
        
        Args:
            video_bytes: Video file content as bytes
            
        Returns:
            Dictionary with prediction results (same as predict())
            
        Raises:
            ValueError: If video processing fails
        """
        # Save bytes to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Run prediction on temporary file
            result = self.predict(tmp_path)
            return result
        finally:
            # Clean up temporary file
            try:
                os.remove(tmp_path)
            except:
                pass  # Ignore cleanup errors
