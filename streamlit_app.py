
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
import tempfile
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model as KModel
from tensorflow.keras.applications import EfficientNetB0
from PIL import Image
import mediapipe as mp

st.set_page_config(
    page_title="Sign Language Classifier",
    page_icon="🤟",
    layout="wide"
)

st.title("🤟 Sign Language Classifier")
st.markdown("Classifies sign language videos into ALPHABET, NUMBERS, or UNIQUE WORDS")

@st.cache_resource
def load_models():
    model = tf.keras.models.load_model('real_lstm_final.keras')
    base = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
    extractor = KModel(inputs=base.input, outputs=base.output)
    return model, extractor

try:
    model, extractor = load_models()
    class_names = ['ALPHABET', 'NUMBERS', 'UNIQUE WORDS']
    SEQ_LEN = 8
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

def extract_features(video_file, seq_len=8):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        video_path = tmp_file.name
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < seq_len:
        cap.release()
        return None
    
    indices = np.linspace(0, total_frames - 1, seq_len, dtype=int)
    frames = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    Path(video_path).unlink()
    
    if len(frames) < seq_len:
        return None
    
    batch = preprocess_input(np.array(frames, dtype='float32'))
    features = extractor.predict(batch, verbose=0)
    return features

def predict_video(video_file):
    features = extract_features(video_file)
    if features is None:
        return None
    
    features = np.expand_dims(features, axis=0)
    predictions = model.predict(features, verbose=0)[0]
    class_id = np.argmax(predictions)
    confidence = predictions[class_id]
    
    return {
        'class': class_names[class_id],
        'confidence': float(confidence),
        'probabilities': {
            'ALPHABET': float(predictions[0]),
            'NUMBERS': float(predictions[1]),
            'UNIQUE WORDS': float(predictions[2])
        }
    }

with st.sidebar:
    st.header("Instructions")
    st.markdown("1. Upload a video file\n2. Wait for processing\n3. View results")
    st.header("Classes")
    st.markdown("- ALPHABET: Letters A-Z\n- NUMBERS: 0-20\n- UNIQUE WORDS: Medical terms")

uploaded_file = st.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.video(uploaded_file)
    
    if st.button("Classify", type="primary"):
        with st.spinner("Processing..."):
            result = predict_video(uploaded_file)
        
        if result:
            with col2:
                st.success(f"### {result['class']}")
                st.metric("Confidence", f"{result['confidence']:.2%}")
                for name, prob in result['probabilities'].items():
                    st.write(f"**{name}**")
                    st.progress(prob, text=f"{prob:.2%}")
        else:
            st.error("Could not process video")
