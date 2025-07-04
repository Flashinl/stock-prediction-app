# ASL-to-Text AI Model

An advanced AI system for translating American Sign Language (ASL) into written text with real-time and pre-recorded video processing capabilities.

## 🎯 Core Objective

This AI model specializes in translating American Sign Language (ASL) into written text, powering a web application that provides real-time and pre-recorded ASL translation services with exceptional accuracy and minimal latency.

## ✨ Key Features

### Extreme Accuracy & Nuance
- Near-zero error rate translation
- Recognition of subtle handshape, movement, and facial expression variations
- Extensive vocabulary including technical jargon, slang, and regional dialects
- Contextual understanding for ambiguous signs

### High-Speed Real-Time Processing
- Minimal latency between signing and text output
- Consistent performance with high-quality video input
- Seamless conversational flow

### Dual Input Modalities
- **Live Video Stream**: Real-time camera feed processing
- **Pre-recorded Videos**: Upload and transcribe MP4, MOV, AVI files

## 🏗️ Project Structure

```
asl-to-text-ai/
├── src/
│   ├── models/
│   │   ├── asl_detector.py          # Core ASL detection model
│   │   ├── sign_classifier.py       # Sign classification and recognition
│   │   ├── gesture_tracker.py       # Hand and body tracking
│   │   └── context_analyzer.py      # Contextual understanding
│   ├── preprocessing/
│   │   ├── video_processor.py       # Video frame processing
│   │   ├── hand_extractor.py        # Hand region extraction
│   │   └── feature_extractor.py     # Feature extraction from frames
│   ├── translation/
│   │   ├── asl_translator.py        # Main translation engine
│   │   ├── vocabulary.py            # ASL vocabulary database
│   │   └── grammar_processor.py     # ASL grammar to English conversion
│   └── utils/
│       ├── video_utils.py           # Video handling utilities
│       ├── model_utils.py           # Model loading and management
│       └── config.py                # Configuration settings
├── web_app/
│   ├── app.py                       # Flask/FastAPI web application
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   └── assets/
│   ├── templates/
│   │   ├── index.html               # Main interface
│   │   ├── live_translation.html    # Live video translation
│   │   └── upload_translation.html  # File upload translation
│   └── api/
│       ├── translation_api.py       # REST API endpoints
│       └── websocket_handler.py     # Real-time WebSocket handling
├── data/
│   ├── datasets/
│   ├── models/                      # Trained model files
│   └── vocabulary/                  # ASL vocabulary data
├── tests/
│   ├── test_models.py
│   ├── test_translation.py
│   └── test_api.py
├── docs/
│   ├── API.md
│   ├── DEPLOYMENT.md
│   └── TRAINING.md
├── requirements.txt
├── setup.py
├── Dockerfile
└── docker-compose.yml
```

## 🚀 Quick Start

### Deploy to Google Cloud Run (Recommended - FREE with 2GB RAM)
```bash
git clone https://github.com/Flashinl/asl_ai.git
cd asl_ai
gcloud run deploy asl-to-text-ai --source . --region us-central1 --memory 2Gi --allow-unauthenticated
```

**See [DEPLOY_CLOUD_RUN.md](DEPLOY_CLOUD_RUN.md) for detailed instructions**

### Local Development

#### Prerequisites
- Python 3.8+
- OpenCV
- TensorFlow
- MediaPipe
- Flask

#### Installation
```bash
git clone https://github.com/Flashinl/asl_ai.git
cd asl_ai
pip install -r requirements.txt
```

#### Running the Application
```bash
python web_app/app.py
```

Access at: http://localhost:5000

## 🔧 Technical Specifications

### Input
- Video stream or video file containing ASL communication
- Supported formats: MP4, MOV, AVI, WebM
- Minimum resolution: 720p for optimal accuracy

### Output
- Clean, formatted text string representing signed communication
- Real-time text display with minimal latency
- Confidence scores for each translation

### Error Handling
- Graceful handling of unclear signs with `[unintelligible sign]` markers
- Contextual disambiguation for multiple-meaning signs
- Quality indicators for translation confidence

## 🎯 Performance Goals

- **Accuracy**: >95% for common vocabulary, >90% for technical terms
- **Latency**: <200ms for real-time translation
- **Throughput**: Process 30+ FPS video streams
- **Vocabulary**: 10,000+ ASL signs and phrases

## 🤝 Contributing

Please read our contributing guidelines and code of conduct before submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support, email support@asl-ai.com or join our Discord community.
