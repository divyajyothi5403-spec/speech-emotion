// Speech Emotion Recognition Website - Interactive JavaScript

// DOM Elements
const startRecordingBtn = document.getElementById('startRecording');
const stopRecordingBtn = document.getElementById('stopRecording');
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');
const audioCanvas = document.getElementById('audioCanvas');
const ctx = audioCanvas.getContext('2d');

// File upload elements
const uploadArea = document.getElementById('uploadArea');
const audioFileInput = document.getElementById('audioFileInput');
const browseBtn = document.getElementById('browseBtn');
const filePreview = document.getElementById('filePreview');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const audioPreview = document.getElementById('audioPreview');
const removeFileBtn = document.getElementById('removeFile');
const analyzeFileBtn = document.getElementById('analyzeFile');
const uploadStatusIndicator = document.getElementById('uploadStatusIndicator');
const uploadStatusText = document.getElementById('uploadStatusText');
const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

// Emotion display elements
const emotionIcon = document.getElementById('emotionIcon');
const emotionLabel = document.getElementById('emotionLabel');
const emotionConfidence = document.getElementById('emotionConfidence');

// Probability bars
const happyProb = document.getElementById('happyProb');
const happyProbText = document.getElementById('happyProbText');
const sadProb = document.getElementById('sadProb');
const sadProbText = document.getElementById('sadProbText');
const angryProb = document.getElementById('angryProb');
const angryProbText = document.getElementById('angryProbText');
const neutralProb = document.getElementById('neutralProb');
const neutralProbText = document.getElementById('neutralProbText');
const probabilityRows = document.querySelectorAll('.prob-bar[data-emotion]');

// Audio recording variables
let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let audioContext;
let analyser;
let dataArray;
let animationId;

// File upload variables
let uploadedFile = null;

// Emotion icons mapping
const emotionIcons = {
    happy: '😊',
    sad: '😢',
    angry: '😠',
    neutral: '😐',
    fear: '😨',
    disgust: '🤢',
    surprise: '😲'
};

// Simulated emotion predictions (for demo purposes)
const simulatedPredictions = [
    { emotion: 'happy', confidence: 0.85, probabilities: { happy: 85, sad: 5, angry: 3, neutral: 7 } },
    { emotion: 'sad', confidence: 0.78, probabilities: { happy: 8, sad: 78, angry: 5, neutral: 9 } },
    { emotion: 'angry', confidence: 0.92, probabilities: { happy: 3, sad: 2, angry: 92, neutral: 3 } },
    { emotion: 'neutral', confidence: 0.88, probabilities: { happy: 5, sad: 4, angry: 3, neutral: 88 } },
    { emotion: 'happy', confidence: 0.91, probabilities: { happy: 91, sad: 3, angry: 2, neutral: 4 } }
];

let predictionIndex = 0;

// Initialize audio canvas
function initAudioCanvas() {
    audioCanvas.width = audioCanvas.offsetWidth;
    audioCanvas.height = audioCanvas.offsetHeight;
    drawIdleWaveform();
}

// Draw idle waveform
function drawIdleWaveform() {
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, audioCanvas.width, audioCanvas.height);
    
    ctx.strokeStyle = '#00d4ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    const centerY = audioCanvas.height / 2;
    const amplitude = 20;
    
    for (let x = 0; x < audioCanvas.width; x++) {
        const y = centerY + Math.sin(x * 0.05) * amplitude;
        if (x === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    
    ctx.stroke();
}

// Draw audio visualization
function drawAudioVisualization() {
    if (!isRecording) {
        drawIdleWaveform();
        return;
    }
    
    analyser.getByteTimeDomainData(dataArray);
    
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, audioCanvas.width, audioCanvas.height);
    
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#00d4ff';
    ctx.shadowBlur = 10;
    ctx.shadowColor = '#00d4ff';
    
    ctx.beginPath();
    
    const sliceWidth = audioCanvas.width / dataArray.length;
    let x = 0;
    
    for (let i = 0; i < dataArray.length; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * audioCanvas.height) / 2;
        
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
        
        x += sliceWidth;
    }
    
    ctx.lineTo(audioCanvas.width, audioCanvas.height / 2);
    ctx.stroke();
    
    ctx.shadowBlur = 0;
    
    animationId = requestAnimationFrame(drawAudioVisualization);
}

// Start recording
async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        const source = audioContext.createMediaStreamSource(stream);
        
        analyser.fftSize = 2048;
        const bufferLength = analyser.frequencyBinCount;
        dataArray = new Uint8Array(bufferLength);
        
        source.connect(analyser);
        
        const recorderMimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
            ? 'audio/webm;codecs=opus'
            : '';
        mediaRecorder = new MediaRecorder(stream, recorderMimeType ? { mimeType: recorderMimeType } : undefined);
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
            const extension = audioBlob.type.includes('webm') ? 'webm' : 'wav';
            await processAudio(audioBlob, `recording.${extension}`);
            
            // Stop all tracks
            stream.getTracks().forEach(track => track.stop());
            audioContext.close();
        };
        
        mediaRecorder.start();
        isRecording = true;
        
        // Update UI
        startRecordingBtn.disabled = true;
        stopRecordingBtn.disabled = false;
        statusIndicator.classList.add('recording');
        statusText.textContent = 'Recording...';
        
        // Start visualization
        drawAudioVisualization();
        
    } catch (error) {
        console.error('Error accessing microphone:', error);
        statusText.textContent = 'Error: Could not access microphone';
        statusIndicator.style.background = '#ff6b6b';
    }
}

// Stop recording
function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        
        // Update UI
        startRecordingBtn.disabled = false;
        stopRecordingBtn.disabled = true;
        statusIndicator.classList.remove('recording');
        statusIndicator.classList.add('processing');
        statusText.textContent = 'Processing...';
        
        // Stop visualization
        cancelAnimationFrame(animationId);
    }
}

// Process audio and predict emotion
async function processAudio(audioBlob, filename = 'recording.webm') {
    try {
        // Create form data
        const formData = new FormData();
        formData.append('audio', audioBlob, filename);
        
        // Send to backend
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const result = await response.json();
        
        // Format prediction for display
        const prediction = {
            emotion: result.emotion,
            confidence: result.confidence,
            probabilities: result.probabilities
        };
        
        // Update emotion display
        updateEmotionDisplay(prediction);
        
        // Update UI
        statusIndicator.classList.remove('processing');
        statusText.textContent = 'Analysis complete';
        
        // Reset after delay
        setTimeout(() => {
            statusText.textContent = 'Ready to record';
            drawIdleWaveform();
        }, 3000);
        
    } catch (error) {
        console.error('Error processing audio:', error);
        
        // Fallback to simulated predictions if backend is not available
        console.log('Using simulated predictions (backend not available)');
        
        // Simulate processing delay
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Get simulated prediction
        const prediction = simulatedPredictions[predictionIndex % simulatedPredictions.length];
        predictionIndex++;
        
        // Update emotion display
        updateEmotionDisplay(prediction);
        
        // Update UI
        statusIndicator.classList.remove('processing');
        statusText.textContent = 'Analysis complete (Demo Mode)';
        
        // Reset after delay
        setTimeout(() => {
            statusText.textContent = 'Ready to record';
            drawIdleWaveform();
        }, 3000);
    }
}

// Update emotion display
function updateEmotionDisplay(prediction) {
    const supportedEmotions = ['happy', 'sad', 'angry', 'neutral'];
    const probabilities = supportedEmotions.reduce((values, emotion) => {
        const rawValue = Number(prediction.probabilities?.[emotion] ?? 0);
        values[emotion] = rawValue <= 1 ? rawValue * 100 : rawValue;
        return values;
    }, {});
    const highestEmotion = supportedEmotions.reduce((winner, emotion) => (
        probabilities[emotion] > probabilities[winner] ? emotion : winner
    ), supportedEmotions[0]);

    // Update main emotion display
    emotionIcon.textContent = emotionIcons[highestEmotion] || '🎤';
    emotionLabel.textContent = `${highestEmotion.charAt(0).toUpperCase() + highestEmotion.slice(1)} detected`;
    emotionConfidence.textContent = `Highest emotion level: ${probabilities[highestEmotion].toFixed(1)}%`;
    
    // Add bounce animation
    emotionIcon.style.animation = 'none';
    setTimeout(() => {
        emotionIcon.style.animation = 'emotion-bounce 0.5s ease';
    }, 10);
    
    // Update probability bars
    const probabilityElements = {
        happy: [happyProb, happyProbText],
        sad: [sadProb, sadProbText],
        angry: [angryProb, angryProbText],
        neutral: [neutralProb, neutralProbText]
    };

    probabilityRows.forEach((row) => {
        row.classList.toggle('highest', row.dataset.emotion === highestEmotion);
    });

    supportedEmotions.forEach((emotion) => {
        const [bar, text] = probabilityElements[emotion];
        const value = Math.min(100, Math.max(0, probabilities[emotion]));
        bar.style.width = `${value}%`;
        text.textContent = `${value.toFixed(1)}%`;
    });
}

// Handle file upload
function handleFileUpload(file) {
    // Check if file is audio
    if (!file.type.startsWith('audio/')) {
        alert('Please upload an audio file.');
        return;
    }
    
    uploadedFile = file;
    
    // Update file preview
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    
    // Create audio preview URL
    const audioUrl = URL.createObjectURL(file);
    audioPreview.src = audioUrl;
    
    // Show file preview, hide upload area
    uploadArea.classList.add('hidden');
    filePreview.classList.remove('hidden');
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Reset file upload
function resetFileUpload() {
    uploadedFile = null;
    audioFileInput.value = '';
    audioPreview.src = '';
    
    uploadArea.classList.remove('hidden');
    filePreview.classList.add('hidden');
    
    // Reset emotion display
    resetEmotionDisplay();
}

// Analyze uploaded file
async function analyzeUploadedFile() {
    if (!uploadedFile) return;
    
    // Update status
    if (uploadStatusIndicator) {
        uploadStatusIndicator.classList.add('processing');
    }
    if (uploadStatusText) {
        uploadStatusText.textContent = 'Analyzing file...';
    }
    
    try {
        // Create form data
        const formData = new FormData();
        formData.append('audio', uploadedFile);
        
        // Send to backend
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const result = await response.json();
        
        // Format prediction for display
        const prediction = {
            emotion: result.emotion,
            confidence: result.confidence,
            probabilities: result.probabilities
        };
        
        // Update emotion display
        updateEmotionDisplay(prediction);
        
        // Update status
        if (uploadStatusIndicator) {
            uploadStatusIndicator.classList.remove('processing');
        }
        if (uploadStatusText) {
            uploadStatusText.textContent = 'Analysis complete';
        }
        
        // Reset after delay
        setTimeout(() => {
            if (uploadStatusText) {
                uploadStatusText.textContent = 'Ready to analyze';
            }
        }, 3000);
        
    } catch (error) {
        console.error('Error analyzing file:', error);
        
        // Fallback to simulated predictions
        console.log('Using simulated predictions (backend not available)');
        
        // Simulate processing delay
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Get simulated prediction
        const prediction = simulatedPredictions[predictionIndex % simulatedPredictions.length];
        predictionIndex++;
        
        // Update emotion display
        updateEmotionDisplay(prediction);
        
        // Update status
        if (uploadStatusIndicator) {
            uploadStatusIndicator.classList.remove('processing');
        }
        if (uploadStatusText) {
            uploadStatusText.textContent = 'Analysis complete (Demo Mode)';
        }
        
        // Reset after delay
        setTimeout(() => {
            if (uploadStatusText) {
                uploadStatusText.textContent = 'Ready to analyze';
            }
        }, 3000);
    }
}

// Reset demo state
function resetDemoState() {
    // Stop recording if active
    if (isRecording) {
        stopRecording();
    }
    
    // Reset recording status
    statusIndicator.classList.remove('recording', 'processing');
    statusText.textContent = 'Ready to record';
    
    // Reset upload status
    if (uploadStatusIndicator) {
        uploadStatusIndicator.classList.remove('processing');
        uploadStatusText.textContent = 'Ready to upload';
    }
    
    // Reset emotion display
    resetEmotionDisplay();
    
    // Reset audio visualization
    if (!isRecording) {
        drawIdleWaveform();
    }
}

// Reset emotion display
function resetEmotionDisplay() {
    emotionIcon.textContent = '🎤';
    emotionLabel.textContent = 'Waiting for input...';
    emotionConfidence.textContent = '--';
    
    // Reset probability bars
    happyProb.style.width = '0%';
    happyProbText.textContent = '0%';
    
    sadProb.style.width = '0%';
    sadProbText.textContent = '0%';
    
    angryProb.style.width = '0%';
    angryProbText.textContent = '0%';
    
    neutralProb.style.width = '0%';
    neutralProbText.textContent = '0%';
    probabilityRows.forEach((row) => row.classList.remove('highest'));
}

// Event listeners
startRecordingBtn.addEventListener('click', startRecording);
stopRecordingBtn.addEventListener('click', stopRecording);

// Tab switching
tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const tabId = btn.dataset.tab;
        
        // Update active tab button
        tabBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        // Update active tab content
        tabContents.forEach(content => {
            content.classList.remove('active');
            if (content.id === tabId + 'Tab') {
                content.classList.add('active');
            }
        });
        
        // Reset status when switching tabs
        resetDemoState();
    });
});

// File upload events
browseBtn.addEventListener('click', () => audioFileInput.click());

uploadArea.addEventListener('click', () => audioFileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileUpload(files[0]);
    }
});

audioFileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
    }
});

removeFileBtn.addEventListener('click', () => {
    resetFileUpload();
});

analyzeFileBtn.addEventListener('click', () => {
    if (uploadedFile) {
        analyzeUploadedFile();
    }
});

// Smooth scrolling for navigation
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Initialize on page load
window.addEventListener('load', () => {
    initAudioCanvas();
});

// Handle window resize
window.addEventListener('resize', () => {
    audioCanvas.width = audioCanvas.offsetWidth;
    audioCanvas.height = audioCanvas.offsetHeight;
    if (!isRecording) {
        drawIdleWaveform();
    }
});

// Add hover effects to process steps
document.querySelectorAll('.process-step').forEach(step => {
    step.addEventListener('mouseenter', () => {
        step.style.transform = 'translateY(-5px)';
    });
    
    step.addEventListener('mouseleave', () => {
        step.style.transform = 'translateY(0)';
    });
});

// Animate elements on scroll
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe process steps
document.querySelectorAll('.process-step').forEach(step => {
    step.style.opacity = '0';
    step.style.transform = 'translateY(30px)';
    step.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(step);
});

// Observe about cards
document.querySelectorAll('.about-card').forEach(card => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(30px)';
    card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(card);
});

// Add typing effect to hero title
function typeWriter(element, text, speed = 100) {
    let i = 0;
    element.textContent = '';
    
    function type() {
        if (i < text.length) {
            element.textContent += text.charAt(i);
            i++;
            setTimeout(type, speed);
        }
    }
    
    type();
}

// Initialize typing effect on page load
window.addEventListener('load', () => {
    const heroTitle = document.querySelector('.hero-title .glow-text');
    if (heroTitle) {
        const originalText = heroTitle.textContent;
        typeWriter(heroTitle, originalText, 150);
    }
});

// Add parallax effect to hero section
window.addEventListener('scroll', () => {
    const scrolled = window.pageYOffset;
    const hero = document.querySelector('.hero');
    const heroVisual = document.querySelector('.hero-visual');
    
    if (hero && heroVisual) {
        heroVisual.style.transform = `translateY(${scrolled * 0.3}px)`;
    }
});

// Add navbar background on scroll
window.addEventListener('scroll', () => {
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 50) {
        navbar.style.background = 'rgba(5, 5, 16, 0.98)';
    } else {
        navbar.style.background = 'rgba(10, 10, 26, 0.95)';
    }
});

// Add click effect to buttons
document.querySelectorAll('.btn').forEach(button => {
    button.addEventListener('click', function(e) {
        const ripple = document.createElement('span');
        const rect = this.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        const x = e.clientX - rect.left - size / 2;
        const y = e.clientY - rect.top - size / 2;
        
        ripple.style.width = ripple.style.height = `${size}px`;
        ripple.style.left = `${x}px`;
        ripple.style.top = `${y}px`;
        ripple.style.position = 'absolute';
        ripple.style.borderRadius = '50%';
        ripple.style.background = 'rgba(255, 255, 255, 0.3)';
        ripple.style.transform = 'scale(0)';
        ripple.style.animation = 'ripple 0.6s ease-out';
        
        this.style.position = 'relative';
        this.style.overflow = 'hidden';
        this.appendChild(ripple);
        
        setTimeout(() => ripple.remove(), 600);
    });
});

// Add ripple animation
const style = document.createElement('style');
style.textContent = `
    @keyframes ripple {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Add audio bars animation
function animateAudioBars() {
    const bars = document.querySelectorAll('.audio-bars .bar');
    bars.forEach(bar => {
        const randomHeight = Math.random() * 80 + 20;
        bar.style.height = `${randomHeight}%`;
    });
}

// Animate audio bars periodically
setInterval(animateAudioBars, 200);

// Add LSTM gate hover effects
document.querySelectorAll('.gate').forEach(gate => {
    gate.addEventListener('mouseenter', () => {
        gate.style.transform = 'scale(1.05)';
        gate.style.boxShadow = '0 0 20px rgba(155, 89, 182, 0.5)';
    });
    
    gate.addEventListener('mouseleave', () => {
        gate.style.transform = 'scale(1)';
        gate.style.boxShadow = 'none';
    });
});

// Add chart bar animation on scroll
const chartObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const bars = entry.target.querySelectorAll('.chart-bar');
            bars.forEach((bar, index) => {
                setTimeout(() => {
                    const value = bar.dataset.value;
                    bar.querySelector('.bar-fill').style.width = `${value}%`;
                }, index * 200);
            });
        }
    });
}, { threshold: 0.5 });

const emotionChart = document.querySelector('.emotion-chart');
if (emotionChart) {
    chartObserver.observe(emotionChart);
}

// Console welcome message
console.log('%c🎤 Speech Emotion Recognition', 'color: #00d4ff; font-size: 20px; font-weight: bold;');
console.log('%cPowered by RNN/LSTM Deep Learning', 'color: #9b59b6; font-size: 14px;');