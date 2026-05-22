<div align="center">

# 🎭 Emotion Detector

### 😠 🤢 😨 😊 😐 😞 😲

**Real-Time Facial Emotion Recognition powered by Deep Learning**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logoColor=white)](https://emotion-detector-dev-by-arif.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-MArifBrohi-181717?style=for-the-badge&logo=github)](https://github.com/MArifBrohi/emotion-detector)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

</div>

---

# ✨ About The Project

Emotion Detector is a Deep Learning based web application that detects human facial emotions in real time from images or webcam input.

The model analyzes facial expressions and predicts one of the following emotions:

- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😊 Happy
- 😐 Neutral
- 😞 Sad
- 😲 Surprise

---

# 🚀 Live Demo

### 🔗 Try Here:
https://emotion-detector-dev-by-arif.streamlit.app/

---

# 🧠 Model Architecture

Custom CNN (Convolutional Neural Network)

```text
Input Image (48x48 grayscale)
        │
        ▼
Conv2D + BatchNorm
        ▼
MaxPooling + Dropout
        ▼
Conv2D + BatchNorm
        ▼
MaxPooling + Dropout
        ▼
Conv2D + BatchNorm
        ▼
Flatten
        ▼
Dense Layers
        ▼
Softmax Output (7 Classes)
