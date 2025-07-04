# ASL-to-Text AI Deployment Guide

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Flashinl/asl_ai.git
cd asl_ai
```

### 2. Local Development Setup

#### Prerequisites
- Python 3.8+
- OpenCV
- Camera access for live translation

#### Install Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

#### Run the Application
```bash
python web_app/app.py
```

Access the application at: http://localhost:5000

### 3. Docker Deployment

#### Build and Run with Docker
```bash
# Build the image
docker build -t asl-to-text-ai .

# Run the container
docker run -p 5000:5000 asl-to-text-ai
```

#### Using Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Production Deployment

### Cloud Platforms

#### 1. Render.com Deployment
1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Use these settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python web_app/app.py`
   - **Environment**: Python 3.10

#### 2. Heroku Deployment
```bash
# Install Heroku CLI
# Create Procfile
echo "web: python web_app/app.py" > Procfile

# Deploy
heroku create your-asl-ai-app
git push heroku main
```

#### 3. AWS EC2 Deployment
```bash
# On EC2 instance
sudo apt update
sudo apt install python3-pip nginx

# Clone and setup
git clone https://github.com/Flashinl/asl_ai.git
cd asl_ai
pip3 install -r requirements.txt

# Setup systemd service
sudo cp deployment/asl-ai.service /etc/systemd/system/
sudo systemctl enable asl-ai
sudo systemctl start asl-ai

# Configure Nginx
sudo cp deployment/nginx.conf /etc/nginx/sites-available/asl-ai
sudo ln -s /etc/nginx/sites-available/asl-ai /etc/nginx/sites-enabled/
sudo systemctl restart nginx
```

### Environment Variables

Create a `.env` file for production:

```env
# Application
FLASK_ENV=production
DEBUG=false
SECRET_KEY=your-secret-key-here

# Database
DATABASE_URL=postgresql://user:password@localhost/asl_ai

# Logging
LOG_LEVEL=INFO

# Performance
MAX_WORKERS=4
TIMEOUT=30
```

### Performance Optimization

#### 1. Model Optimization
```python
# Use TensorFlow Lite for mobile deployment
import tensorflow as tf

# Convert model to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model('path/to/model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save optimized model
with open('model_optimized.tflite', 'wb') as f:
    f.write(tflite_model)
```

#### 2. Caching Setup
```python
# Redis caching for vocabulary and models
import redis

redis_client = redis.Redis(
    host='localhost',
    port=6379,
    decode_responses=True
)
```

#### 3. Load Balancing
```nginx
# Nginx load balancer configuration
upstream asl_ai_backend {
    server 127.0.0.1:5000;
    server 127.0.0.1:5001;
    server 127.0.0.1:5002;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://asl_ai_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Security Configuration

### 1. HTTPS Setup
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com
```

### 2. Firewall Configuration
```bash
# UFW setup
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw allow 443
sudo ufw enable
```

### 3. Application Security
```python
# Add to app.py
from flask_talisman import Talisman

# Enable security headers
Talisman(app, force_https=True)

# Rate limiting
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)
```

## Monitoring and Logging

### 1. Application Monitoring
```python
# Add to app.py
import logging
from logging.handlers import RotatingFileHandler

# Setup logging
if not app.debug:
    file_handler = RotatingFileHandler(
        'logs/asl_ai.log', 
        maxBytes=10240000, 
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s'
    ))
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
```

### 2. Performance Monitoring
```bash
# Install monitoring tools
pip install prometheus-client grafana-api

# Setup Prometheus metrics endpoint
from prometheus_client import Counter, Histogram, generate_latest

translation_requests = Counter('translation_requests_total', 'Total translation requests')
translation_duration = Histogram('translation_duration_seconds', 'Translation processing time')
```

## Scaling Considerations

### 1. Horizontal Scaling
- Use multiple application instances behind a load balancer
- Implement session storage in Redis/database
- Use CDN for static assets

### 2. Database Scaling
- Use PostgreSQL with read replicas
- Implement connection pooling
- Cache frequently accessed vocabulary data

### 3. Model Serving
- Use TensorFlow Serving for model deployment
- Implement model versioning
- Use GPU acceleration for better performance

## Troubleshooting

### Common Issues

#### 1. Camera Access Issues
```javascript
// Check browser permissions
navigator.mediaDevices.getUserMedia({video: true})
  .then(stream => console.log('Camera access granted'))
  .catch(err => console.error('Camera access denied:', err));
```

#### 2. WebSocket Connection Issues
```python
# Check CORS settings
from flask_cors import CORS
CORS(app, origins=["*"])

# Enable WebSocket debugging
socketio = SocketIO(app, logger=True, engineio_logger=True)
```

#### 3. Model Loading Issues
```python
# Check model file paths
import os
model_path = 'data/models/asl_model.h5'
if not os.path.exists(model_path):
    print(f"Model file not found: {model_path}")
```

### Performance Issues

#### 1. High Memory Usage
```python
# Implement model caching
import functools

@functools.lru_cache(maxsize=1)
def load_model():
    return tf.keras.models.load_model('model.h5')
```

#### 2. Slow Translation
```python
# Use threading for video processing
import threading
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)
```

## Maintenance

### 1. Regular Updates
```bash
# Update dependencies
pip install --upgrade -r requirements.txt

# Update models
python scripts/update_models.py

# Backup database
pg_dump asl_ai > backup_$(date +%Y%m%d).sql
```

### 2. Health Checks
```python
# Implement health check endpoint
@app.route('/health')
def health_check():
    return {
        'status': 'healthy',
        'timestamp': time.time(),
        'version': '1.0.0'
    }
```

### 3. Log Rotation
```bash
# Setup logrotate
sudo nano /etc/logrotate.d/asl-ai

# Add configuration:
/var/log/asl-ai/*.log {
    daily
    missingok
    rotate 52
    compress
    notifempty
    create 644 www-data www-data
}
```
