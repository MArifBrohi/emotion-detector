import streamlit as st
import numpy as np
import json
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, MaxPooling2D,
    Dropout, Flatten, Dense, Input
)

st.set_page_config(
    page_title="Emotion Detector",
    page_icon="😊",
    layout="centered"
)

st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #f093fb, #f5576c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.developer {
    text-align: center;
    color: #f5576c;
    font-size: 0.85rem;
    font-weight: 600;
}
.emotion-box {
    text-align: center;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">😊 Real-Time Emotion Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="developer">👨‍💻 Developed by Muhammad Arif | CNN Deep Learning Model</p>', unsafe_allow_html=True)
st.divider()

@st.cache_resource
def load_emotion_model():
    # ── Build the model architecture manually ──────────────────────────────
    # This avoids the Keras version mismatch caused by 'quantization_config'
    # in the saved JSON (produced by a newer Keras that your env doesn't support).
    model = Sequential([
        Input(shape=(48, 48, 1)),

        # Block 1
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(momentum=0.99, epsilon=0.001),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(momentum=0.99, epsilon=0.001),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(momentum=0.99, epsilon=0.001),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(momentum=0.99, epsilon=0.001),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(momentum=0.99, epsilon=0.001),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(momentum=0.99, epsilon=0.001),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Dropout(0.25),

        # Classifier head
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(momentum=0.99, epsilon=0.001),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(momentum=0.99, epsilon=0.001),
        Dropout(0.5),
        Dense(7, activation='softmax'),
    ])

    # ── Load weights ───────────────────────────────────────────────────────
    model.load_weights("emotion_weights.weights.h5")
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # ── Load class labels ──────────────────────────────────────────────────
    with open("class_labels.json", "r") as f:
        labels = json.load(f)

    return model, labels

model, class_labels = load_emotion_model()

emotion_emoji = {
    'Angry':    '😠',
    'Disgust':  '🤢',
    'Fear':     '😨',
    'Happy':    '😊',
    'Neutral':  '😐',
    'Sad':      '😞',
    'Surprise': '😲'
}

emotion_colors = {
    'Angry':    '#e74c3c',
    'Disgust':  '#8e44ad',
    'Fear':     '#2c3e50',
    'Happy':    '#f1c40f',
    'Neutral':  '#95a5a6',
    'Sad':      '#3498db',
    'Surprise': '#e67e22'
}

def detect_emotion(image):
    img_array = np.array(image)

    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1,
        minNeighbors=5, minSize=(30, 30)
    )

    results = []

    if len(faces) == 0:
        return None, img_array, "No face detected!"

    for (x, y, w, h) in faces:
        face_roi        = gray[y:y+h, x:x+w]
        face_resized    = cv2.resize(face_roi, (48, 48))
        face_normalized = face_resized / 255.0
        face_input      = face_normalized.reshape(1, 48, 48, 1)

        predictions  = model.predict(face_input, verbose=0)
        emotion_idx  = np.argmax(predictions[0])
        emotion      = class_labels[str(emotion_idx)]
        confidence   = predictions[0][emotion_idx] * 100

        results.append({
            'emotion':     emotion,
            'confidence':  confidence,
            'predictions': predictions[0],
            'bbox':        (x, y, w, h)
        })

        color = (255, 165, 0)
        cv2.rectangle(img_array, (x, y), (x+w, y+h), color, 2)
        label = f"{emotion} ({confidence:.1f}%)"
        cv2.putText(img_array, label, (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return results, img_array, None

tab1, tab2 = st.tabs(["📸 Image Upload", "📷 Webcam"])

with tab1:
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader(
        "Upload a photo with a face",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original Image", use_container_width=True)

        with st.spinner("Detecting emotion..."):
            results, processed_img, error = detect_emotion(image)

        if error:
            st.warning(f"⚠️ {error}")
        else:
            with col2:
                st.image(processed_img, caption="Detected Face", use_container_width=True)

            st.divider()

            for result in results:
                emotion = result['emotion']
                conf    = result['confidence']
                emoji   = emotion_emoji[emotion]
                color   = emotion_colors[emotion]

                st.markdown(f"""
                <div class="emotion-box" style="background:{color}20; border: 2px solid {color};">
                    <h2 style="color:{color}; margin:0;">{emoji} {emotion}</h2>
                    <p style="font-size:1.2rem; margin:0;">Confidence: <b>{conf:.1f}%</b></p>
                </div>
                """, unsafe_allow_html=True)

                st.subheader("All Emotion Probabilities:")
                for idx, prob in enumerate(result['predictions']):
                    emotion_name = class_labels[str(idx)]
                    emoji_icon   = emotion_emoji[emotion_name]
                    st.progress(float(prob), text=f"{emoji_icon} {emotion_name}: {prob*100:.1f}%")

with tab2:
    st.subheader("Webcam — Live Emotion Detection")
    st.info("📷 Take a photo using your webcam!")

    webcam_photo = st.camera_input("Take a photo")

    if webcam_photo:
        image = Image.open(webcam_photo).convert('RGB')

        with st.spinner("Detecting emotion..."):
            results, processed_img, error = detect_emotion(image)

        if error:
            st.warning(f"⚠️ {error}")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Your Photo", use_container_width=True)
            with col2:
                st.image(processed_img, caption="Detected Face", use_container_width=True)

            for result in results:
                emotion = result['emotion']
                conf    = result['confidence']
                emoji   = emotion_emoji[emotion]
                color   = emotion_colors[emotion]

                st.markdown(f"""
                <div class="emotion-box" style="background:{color}20; border: 2px solid {color};">
                    <h2 style="color:{color}; margin:0;">{emoji} {emotion}</h2>
                    <p style="font-size:1.2rem; margin:0;">Confidence: <b>{conf:.1f}%</b></p>
                </div>
                """, unsafe_allow_html=True)

                st.subheader("All Emotion Probabilities:")
                for idx, prob in enumerate(result['predictions']):
                    emotion_name = class_labels[str(idx)]
                    emoji_icon   = emotion_emoji[emotion_name]
                    st.progress(float(prob), text=f"{emoji_icon} {emotion_name}: {prob*100:.1f}%")

st.divider()
st.markdown("""
<div style='text-align:center; color:#adb5bd; font-size:0.8rem;'>
    Built with CNN Deep Learning | TensorFlow + OpenCV + Streamlit<br>
    Trained on FER2013 Dataset — 35,000+ facial images, 7 emotions<br>
    Made with ❤️ by <b>Muhammad Arif</b>
</div>
""", unsafe_allow_html=True)
