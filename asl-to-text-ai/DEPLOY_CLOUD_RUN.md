# Deploy ASL-to-Text AI to Google Cloud Run

## 🚀 Why Google Cloud Run?

- **2GB RAM** (vs Render's 512MB free tier)
- **2 CPU cores** available
- **Generous free tier**: 2 million requests/month
- **Auto-scaling**: Scales to zero when not in use
- **Perfect for ML apps**: Handles TensorFlow and OpenCV

## 📋 Prerequisites

1. **Google Cloud Account** (free tier available)
2. **Google Cloud SDK** installed
3. **Docker** installed (optional, Cloud Build handles this)

## 🛠️ Deployment Steps

### Step 1: Set Up Google Cloud

1. **Go to**: https://console.cloud.google.com/
2. **Create a new project** or select existing one
3. **Enable APIs**:
   - Cloud Run API
   - Cloud Build API
   - Container Registry API

### Step 2: Install Google Cloud SDK

**Windows:**
```bash
# Download and install from: https://cloud.google.com/sdk/docs/install
# Or use PowerShell:
(New-Object Net.WebClient).DownloadFile("https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe", "$env:Temp\GoogleCloudSDKInstaller.exe")
& $env:Temp\GoogleCloudSDKInstaller.exe
```

**After installation:**
```bash
# Login to Google Cloud
gcloud auth login

# Set your project
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### Step 3: Deploy from GitHub

**Option A: Direct Deploy (Easiest)**
```bash
# Clone your repository
git clone https://github.com/Flashinl/asl_ai.git
cd asl_ai

# Deploy to Cloud Run
gcloud run deploy asl-to-text-ai \
  --source . \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --port 8080 \
  --allow-unauthenticated \
  --max-instances 10
```

**Option B: Using Cloud Build**
```bash
# Submit build
gcloud builds submit --config cloudbuild.yaml
```

### Step 4: Configure Environment Variables (Optional)

```bash
gcloud run services update asl-to-text-ai \
  --region us-central1 \
  --set-env-vars FLASK_ENV=production,DEBUG=false
```

## 🎯 Expected Results

After deployment:
- **URL**: https://asl-to-text-ai-[hash]-uc.a.run.app
- **Resources**: 2GB RAM, 2 CPU cores
- **Features**: Full ASL translation with ML capabilities
- **Cost**: Free tier covers most usage

## 💰 Cost Estimation

**Free Tier Includes:**
- 2 million requests/month
- 400,000 GB-seconds/month
- 200,000 CPU-seconds/month

**Typical Usage:**
- Small app: **$0/month** (within free tier)
- Medium usage: **$5-15/month**
- Heavy usage: **$20-50/month**

## 🔧 Troubleshooting

### Build Fails - Memory Issues
```bash
# Use Cloud Build with more resources
gcloud builds submit --config cloudbuild.yaml --machine-type=e2-highcpu-8
```

### Cold Start Issues
```bash
# Set minimum instances
gcloud run services update asl-to-text-ai \
  --region us-central1 \
  --min-instances 1
```

### Timeout Issues
```bash
# Increase timeout
gcloud run services update asl-to-text-ai \
  --region us-central1 \
  --timeout 300
```

## 📊 Monitoring

**View logs:**
```bash
gcloud run services logs read asl-to-text-ai --region us-central1
```

**Monitor performance:**
- Go to Cloud Console → Cloud Run → asl-to-text-ai
- Check metrics for CPU, memory, requests

## 🎉 Success!

Your ASL-to-Text AI will be live at:
`https://asl-to-text-ai-[hash]-uc.a.run.app`

Features available:
- ✅ Real-time ASL translation
- ✅ Video upload processing  
- ✅ Full ML pipeline (TensorFlow + MediaPipe)
- ✅ Professional web interface
- ✅ 2GB RAM for smooth performance
