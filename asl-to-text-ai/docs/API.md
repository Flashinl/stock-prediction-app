# ASL-to-Text AI API Documentation

## Overview

The ASL-to-Text AI system provides both REST API endpoints and WebSocket connections for real-time sign language translation.

## Base URL

```
http://localhost:5000
```

## Authentication

Currently, the API does not require authentication. In production, implement proper authentication mechanisms.

## REST API Endpoints

### Health Check

**GET** `/api/health`

Check the health status of the API.

**Response:**
```json
{
  "status": "healthy",
  "translator_ready": true,
  "version": "1.0.0"
}
```

### Upload Video Translation

**POST** `/api/translate/upload`

Translate an uploaded video file.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Form data with `video` file

**Response:**
```json
{
  "success": true,
  "translation": "Hello how are you today",
  "word_count": 5,
  "processing_stats": {
    "total_signs_processed": 15,
    "successful_translations": 5,
    "success_rate": 0.33,
    "average_confidence": 0.85,
    "average_processing_time": 0.045
  },
  "confidence_scores": [0.92, 0.87, 0.83, 0.89, 0.78]
}
```

### Translation Statistics

**GET** `/api/stats`

Get current translation performance statistics.

**Response:**
```json
{
  "total_signs_processed": 150,
  "successful_translations": 142,
  "success_rate": 0.947,
  "average_confidence": 0.863,
  "average_processing_time": 0.042,
  "max_processing_time": 0.089,
  "min_processing_time": 0.021
}
```

### Vocabulary Information

**GET** `/api/vocabulary`

Get vocabulary statistics and information.

**Response:**
```json
{
  "total_words": 1000,
  "total_categories": 15,
  "categories": ["greeting", "courtesy", "response", "action", "noun", "verb"],
  "category_counts": {
    "greeting": 25,
    "courtesy": 18,
    "response": 12
  },
  "most_common_category": "noun"
}
```

**GET** `/api/vocabulary/categories`

Get list of all vocabulary categories.

**Response:**
```json
{
  "categories": ["greeting", "courtesy", "response", "action", "noun", "verb", "pronoun", "time", "number", "family", "color"]
}
```

**GET** `/api/vocabulary/category/{category}`

Get all words in a specific category.

**Response:**
```json
{
  "category": "greeting",
  "words": [
    {
      "class_id": 0,
      "word": "hello",
      "category": "greeting",
      "description": "Common ASL sign for 'hello'",
      "difficulty": "easy",
      "frequency": 0.9
    }
  ]
}
```

## WebSocket Events

Connect to WebSocket at `/socket.io/`

### Client Events

#### `connect`
Establish connection to the server.

#### `start_translation`
Start a real-time translation session.

**Emit:**
```javascript
socket.emit('start_translation');
```

#### `video_frame`
Send a video frame for translation.

**Emit:**
```javascript
socket.emit('video_frame', {
  frame: 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...',
  timestamp: 1640995200000
});
```

#### `stop_translation`
Stop the current translation session.

**Emit:**
```javascript
socket.emit('stop_translation');
```

### Server Events

#### `status`
Connection status message.

**Receive:**
```javascript
socket.on('status', (data) => {
  console.log(data.message); // "Connected to ASL translator"
});
```

#### `translation_started`
Confirmation that translation session has started.

**Receive:**
```javascript
socket.on('translation_started', (data) => {
  console.log(data.message); // "Translation session started"
});
```

#### `translation_result`
Real-time translation results.

**Receive:**
```javascript
socket.on('translation_result', (data) => {
  // data structure:
  {
    timestamp: 1640995200000,
    processing_time: 0.045,
    buffer_size: 25,
    detection_ready: true,
    word: "hello",
    confidence: 0.92,
    sentence_complete: false,
    current_sentence: "hello how are"
  }
});
```

#### `translation_stopped`
Translation session has ended.

**Receive:**
```javascript
socket.on('translation_stopped', (data) => {
  // data structure:
  {
    final_sentence: "Hello how are you today",
    stats: {
      total_signs_processed: 15,
      successful_translations: 5,
      success_rate: 0.33,
      average_confidence: 0.85
    }
  }
});
```

#### `error`
Error message from server.

**Receive:**
```javascript
socket.on('error', (data) => {
  console.error(data.message);
});
```

## Error Codes

| Code | Description |
|------|-------------|
| 400  | Bad Request - Invalid input data |
| 404  | Not Found - Endpoint not found |
| 500  | Internal Server Error - Server processing error |

## Rate Limiting

- Video frame uploads: Maximum 30 frames per second
- File uploads: Maximum 100MB per file
- API requests: Maximum 100 requests per minute per IP

## Example Usage

### JavaScript (Live Translation)

```javascript
// Connect to WebSocket
const socket = io();

// Start translation
socket.emit('start_translation');

// Send video frames
function sendFrame(videoElement) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = 640;
  canvas.height = 480;
  ctx.drawImage(videoElement, 0, 0, 640, 480);
  
  const frameData = canvas.toDataURL('image/jpeg', 0.8);
  socket.emit('video_frame', { frame: frameData });
}

// Receive translations
socket.on('translation_result', (data) => {
  if (data.word) {
    console.log('New word:', data.word, 'Confidence:', data.confidence);
  }
  if (data.current_sentence) {
    document.getElementById('output').textContent = data.current_sentence;
  }
});
```

### Python (File Upload)

```python
import requests

# Upload video file
with open('sign_language_video.mp4', 'rb') as f:
    files = {'video': f}
    response = requests.post('http://localhost:5000/api/translate/upload', files=files)
    
if response.status_code == 200:
    result = response.json()
    print('Translation:', result['translation'])
    print('Confidence scores:', result['confidence_scores'])
else:
    print('Error:', response.json()['error'])
```

### cURL (Health Check)

```bash
curl -X GET http://localhost:5000/api/health
```

## Performance Considerations

- **Frame Rate**: For real-time translation, send frames at 10-30 FPS for optimal performance
- **Image Quality**: Use JPEG compression (70-80% quality) to reduce bandwidth
- **Buffer Management**: The system maintains a 30-frame buffer for sequence analysis
- **Timeout**: WebSocket connections timeout after 30 seconds of inactivity

## Security Notes

- Implement HTTPS in production
- Add authentication and authorization
- Validate all input data
- Implement rate limiting
- Use CORS appropriately
- Sanitize file uploads
