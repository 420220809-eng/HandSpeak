"""
Model Loader Module
Loads the GRU model and associated metadata once at startup
"""
import os
import json
import pickle as pkl
from tensorflow import keras
from typing import Dict, Any


class ModelLoader:
    """
    Handles loading of the trained GRU model and its metadata
    """
    
    def __init__(self, model_path: str = 'models/sign_gru_model.h5'):
        """
        Initialize the model loader
        
        Args:
            model_path: Path to the trained model file (.h5)
        """
        self.model_path = model_path
        self.model = None
        self.label_encoder = None
        self.sign_classes = None
        self.metadata = None
        
    def load(self) -> Dict[str, Any]:
        """
        Load the model, label encoder, sign classes, and metadata
        
        Returns:
            Dictionary containing loaded components
            
        Raises:
            FileNotFoundError: If required files are missing
            Exception: If loading fails
        """
        print(f"Loading model from: {self.model_path}")
        
        # Check if model file exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load the trained model
        try:
            self.model = keras.models.load_model(self.model_path)
            print(f"✓ Model loaded successfully")
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
        
        # Load label encoder
        label_encoder_path = 'data/processed/label_encoder.pkl'
        if not os.path.exists(label_encoder_path):
            raise FileNotFoundError(f"Label encoder not found: {label_encoder_path}")
        
        try:
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pkl.load(f)
            print(f"✓ Label encoder loaded")
        except Exception as e:
            raise Exception(f"Failed to load label encoder: {str(e)}")
        
        # Load sign classes
        sign_classes_path = 'data/processed/sign_classes.pkl'
        if not os.path.exists(sign_classes_path):
            raise FileNotFoundError(f"Sign classes not found: {sign_classes_path}")
        
        try:
            with open(sign_classes_path, 'rb') as f:
                self.sign_classes = pkl.load(f)
            print(f"✓ Sign classes loaded: {len(self.sign_classes)} classes")
        except Exception as e:
            raise Exception(f"Failed to load sign classes: {str(e)}")
        
        # Load metadata
        metadata_path = 'models/gru_model_metadata.json'
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                print(f"✓ Metadata loaded")
            except Exception as e:
                print(f"Warning: Could not load metadata: {str(e)}")
                self.metadata = {}
        else:
            print(f"Warning: Metadata file not found, using defaults")
            self.metadata = {}
        
        print(f"\n{'='*60}")
        print(f"Model loaded successfully!")
        print(f"Sign classes: {self.sign_classes}")
        print(f"{'='*60}\n")
        
        return {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'sign_classes': self.sign_classes,
            'metadata': self.metadata
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"error": "Model not loaded"}
        
        return {
            "model_type": self.metadata.get("model_type", "GRU"),
            "model_version": self.metadata.get("model_version", "unknown"),
            "num_classes": len(self.sign_classes) if self.sign_classes else 0,
            "sign_classes": self.sign_classes,
            "sequence_length": self.metadata.get("sequence_length", 30),
            "num_features": self.metadata.get("num_features", 126),
            "confidence_threshold": self.metadata.get("confidence_threshold", 0.7),
            "flutter_compatible": self.metadata.get("flutter_compatible", True)
        }
