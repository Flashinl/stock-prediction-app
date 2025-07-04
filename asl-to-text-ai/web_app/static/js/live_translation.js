/**
 * Live Translation JavaScript for ASL-to-Text AI
 */

// WebSocket connection
const socket = io();

// DOM elements
const videoElement = document.getElementById('videoElement');
const canvasElement = document.getElementById('canvasElement');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const clearBtn = document.getElementById('clearBtn');
const translationOutput = document.getElementById('translationOutput');
const connectionStatus = document.getElementById('connectionStatus');
const connectionText = document.getElementById('connectionText');

// Statistics elements
const currentWordElement = document.getElementById('currentWord');
const currentWordText = document.getElementById('currentWordText');
const currentWordConfidence = document.getElementById('currentWordConfidence');
const confidenceBar = document.getElementById('confidenceBar');
const wordsDetected = document.getElementById('wordsDetected');
const avgConfidence = document.getElementById('avgConfidence');
const processingTime = document.getElementById('processingTime');
const bufferStatus = document.getElementById('bufferStatus');

// State variables
let isTranslating = false;
let stream = null;
let frameInterval = null;
let sessionStats = {
    wordsDetected: 0,
    totalConfidence: 0,
    avgProcessingTime: 0,
    processingTimes: []
};
let translatedText = '';

/**
 * Initialize camera and setup video stream
 */
async function initCamera() {
    try {
        const constraints = {
            video: { 
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: 30 }
            },
            audio: false
        };
        
        stream = await navigator.mediaDevices.getUserMedia(constraints);
        videoElement.srcObject = stream;
        
        // Setup canvas for frame capture
        canvasElement.width = 640;
        canvasElement.height = 480;
        
        console.log('Camera initialized successfully');
        
        // Enable start button once camera is ready
        videoElement.addEventListener('loadedmetadata', () => {
            startBtn.disabled = false;
        });
        
    } catch (error) {
        console.error('Error accessing camera:', error);
        showError('Could not access camera. Please check permissions and try again.');
    }
}

/**
 * Capture and send video frame to server
 */
function sendFrame() {
    if (!isTranslating || !stream || !videoElement.videoWidth) return;
    
    try {
        const context = canvasElement.getContext('2d');
        context.drawImage(videoElement, 0, 0, 640, 480);
        
        // Convert to base64 with compression
        const frameData = canvasElement.toDataURL('image/jpeg', 0.7);
        
        // Send to server
        socket.emit('video_frame', { 
            frame: frameData,
            timestamp: Date.now()
        });
        
    } catch (error) {
        console.error('Error capturing frame:', error);
    }
}

/**
 * Start the translation session
 */
function startTranslation() {
    if (!stream) {
        showError('Camera not initialized. Please refresh the page.');
        return;
    }
    
    isTranslating = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    clearBtn.disabled = true;
    
    updateConnectionStatus('processing');
    socket.emit('start_translation');
    
    // Start sending frames at 10 FPS
    frameInterval = setInterval(sendFrame, 100);
    
    console.log('Translation started');
}

/**
 * Stop the translation session
 */
function stopTranslation() {
    isTranslating = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    clearBtn.disabled = false;
    
    if (frameInterval) {
        clearInterval(frameInterval);
        frameInterval = null;
    }
    
    updateConnectionStatus('connected');
    socket.emit('stop_translation');
    
    console.log('Translation stopped');
}

/**
 * Clear translation output and reset statistics
 */
function clearTranslation() {
    translatedText = '';
    updateTranslationOutput('');
    resetSessionStats();
    hideCurrentWord();
    confidenceBar.style.width = '0%';
    
    console.log('Translation cleared');
}

/**
 * Reset session statistics
 */
function resetSessionStats() {
    sessionStats = {
        wordsDetected: 0,
        totalConfidence: 0,
        avgProcessingTime: 0,
        processingTimes: []
    };
    
    wordsDetected.textContent = '0';
    avgConfidence.textContent = '0%';
    processingTime.textContent = '0ms';
    bufferStatus.textContent = '0/30';
}

/**
 * Socket event handlers
 */
socket.on('connect', () => {
    console.log('Connected to server');
    updateConnectionStatus('connected');
});

socket.on('disconnect', () => {
    console.log('Disconnected from server');
    updateConnectionStatus('disconnected');
    
    if (isTranslating) {
        stopTranslation();
    }
});

socket.on('translation_result', (data) => {
    updateStats(data);
    
    if (data.word) {
        addWordToTranslation(data.word, data.confidence);
        showCurrentWord(data.word, data.confidence);
    }
    
    if (data.processed_sentence) {
        updateTranslationOutput(data.processed_sentence);
    } else if (data.current_sentence) {
        updateTranslationOutput(data.current_sentence);
    }
});

socket.on('translation_started', (data) => {
    console.log('Translation session started on server');
    showSuccess('Translation session started successfully');
});

socket.on('translation_stopped', (data) => {
    console.log('Translation session stopped on server');
    
    if (data.final_sentence) {
        updateTranslationOutput(data.final_sentence);
    }
    
    if (data.stats) {
        console.log('Final session stats:', data.stats);
    }
});

socket.on('error', (data) => {
    console.error('Server error:', data.message);
    showError('Server error: ' + data.message);
    
    if (isTranslating) {
        stopTranslation();
    }
});

/**
 * UI update functions
 */
function updateConnectionStatus(status) {
    connectionStatus.className = `status-indicator status-${status}`;
    
    const statusText = {
        'connected': 'Connected',
        'processing': 'Processing',
        'disconnected': 'Disconnected'
    };
    
    connectionText.textContent = statusText[status] || 'Unknown';
}

function updateStats(data) {
    // Update processing time
    if (data.processing_time) {
        const timeMs = Math.round(data.processing_time * 1000);
        processingTime.textContent = timeMs + 'ms';
        
        sessionStats.processingTimes.push(timeMs);
        if (sessionStats.processingTimes.length > 100) {
            sessionStats.processingTimes.shift();
        }
        
        const avgTime = sessionStats.processingTimes.reduce((a, b) => a + b, 0) / sessionStats.processingTimes.length;
        sessionStats.avgProcessingTime = Math.round(avgTime);
    }
    
    // Update buffer status
    if (data.buffer_size !== undefined) {
        bufferStatus.textContent = `${data.buffer_size}/30`;
    }
    
    // Update confidence bar
    if (data.confidence) {
        const confidence = Math.round(data.confidence * 100);
        confidenceBar.style.width = confidence + '%';
    }
}

function addWordToTranslation(word, confidence) {
    sessionStats.wordsDetected++;
    sessionStats.totalConfidence += confidence;
    
    // Update display
    wordsDetected.textContent = sessionStats.wordsDetected;
    
    const avgConf = (sessionStats.totalConfidence / sessionStats.wordsDetected) * 100;
    avgConfidence.textContent = Math.round(avgConf) + '%';
}

function showCurrentWord(word, confidence) {
    currentWordText.textContent = word;
    currentWordConfidence.textContent = Math.round(confidence * 100);
    currentWordElement.classList.remove('d-none');
    
    // Auto-hide after 3 seconds
    setTimeout(() => {
        hideCurrentWord();
    }, 3000);
}

function hideCurrentWord() {
    currentWordElement.classList.add('d-none');
}

function updateTranslationOutput(text) {
    translatedText = text;
    
    if (text && text.trim()) {
        translationOutput.innerHTML = `<p class="mb-0">${text}</p>`;
    } else {
        translationOutput.innerHTML = `
            <p class="text-muted text-center mb-0">
                <i class="fas fa-hand-paper fa-2x mb-2"></i><br>
                Start signing to see translation here
            </p>
        `;
    }
}

/**
 * Notification functions
 */
function showError(message) {
    // Create toast notification for errors
    const toast = createToast('error', message);
    showToast(toast);
}

function showSuccess(message) {
    // Create toast notification for success
    const toast = createToast('success', message);
    showToast(toast);
}

function createToast(type, message) {
    const toastContainer = getOrCreateToastContainer();
    
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type === 'error' ? 'danger' : 'success'} border-0`;
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                <i class="fas fa-${type === 'error' ? 'exclamation-triangle' : 'check-circle'} me-2"></i>
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    
    toastContainer.appendChild(toast);
    return toast;
}

function getOrCreateToastContainer() {
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container position-fixed top-0 end-0 p-3';
        container.style.zIndex = '9999';
        document.body.appendChild(container);
    }
    return container;
}

function showToast(toastElement) {
    const toast = new bootstrap.Toast(toastElement, {
        autohide: true,
        delay: 5000
    });
    toast.show();
    
    // Remove from DOM after hiding
    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });
}

/**
 * Event listeners
 */
startBtn.addEventListener('click', startTranslation);
stopBtn.addEventListener('click', stopTranslation);
clearBtn.addEventListener('click', clearTranslation);

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden && isTranslating) {
        console.log('Page hidden, pausing translation');
        // Optionally pause translation when page is hidden
    } else if (!document.hidden && isTranslating) {
        console.log('Page visible, resuming translation');
        // Resume translation if needed
    }
});

// Handle beforeunload to cleanup
window.addEventListener('beforeunload', () => {
    if (isTranslating) {
        stopTranslation();
    }
    
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
});

/**
 * Initialize on page load
 */
window.addEventListener('load', () => {
    console.log('Page loaded, initializing camera...');
    initCamera();
    updateConnectionStatus('disconnected');
    resetSessionStats();
    
    // Disable start button until camera is ready
    startBtn.disabled = true;
});
