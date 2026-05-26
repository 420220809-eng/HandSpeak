---
title: HandSpeak API
emoji: 🤟
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# HandSpeak - Arabic Sign Language Recognition API

This is a FastAPI-based REST API for Arabic Sign Language Recognition using GRU (Gated Recurrent Unit) neural networks and MediaPipe for hand landmark detection.

## API Endpoints

- **GET /** - API information and available endpoints
- **GET /api/health** - Health check endpoint
- **GET /api/model-info** - Get model information and sign classes
- **POST /api/predict** - Upload video for sign language prediction
- **GET /api/docs** - Interactive API documentation (Swagger UI)

## Usage

### Health Check
```bash
curl https://daniel-10-handspeak-api.hf.space/api/health
```

### Predict Sign Language
```bash
curl -X POST "https://daniel-10-handspeak-api.hf.space/api/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_video.mp4"
```

### Interactive Documentation
Visit: https://daniel-10-handspeak-api.hf.space/api/docs

## Model Information

- **Architecture**: GRU (Gated Recurrent Unit)
- **Input**: Video files (mp4, mov, avi, mkv, webm)
- **Output**: Predicted Arabic sign with confidence score
- **Supported Signs**: 24 Arabic sign language gestures

## Technical Stack

- **Framework**: FastAPI
- **Deep Learning**: TensorFlow/Keras
- **Computer Vision**: MediaPipe, OpenCV
- **Deployment**: Docker on Hugging Face Spaces

## Response Format

```json
{
  "success": true,
  "predicted_sign": "السلام",
  "confidence": 0.9876,
  "confidence_percentage": 98.76,
  "meets_threshold": true,
  "threshold": 0.7,
  "top_3_predictions": {
    "السلام": 0.9876,
    "عليكم": 0.0089,
    "شكرا": 0.0023
  },
  "extraction_metadata": {
    "total_frames": 45,
    "frames_with_hands": 42,
    "sequence_length": 30
  }
}
```

## License

This project is for educational and research purposes.
