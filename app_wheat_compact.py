import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go

# Constants
CLASS_NAMES = ['Brown Rust', 'Healthy', 'Leaf Blight', 'Mildew', 'Smut', 'Yellow Rust']
MODEL_PATH = "wheat_model_finetuned.h5"

DISEASE_INFO = {
    'Brown Rust': {'severity': '🔴 High', 'treatment': 'Apply fungicides, remove infected plants', 'color': '#EF4444'},
    'Healthy': {'severity': '🟢 None', 'treatment': 'Maintain regular care', 'color': '#10B981'},
    'Leaf Blight': {'severity': '🟡 Medium', 'treatment': 'Use copper-based fungicides', 'color': '#F59E0B'},
    'Mildew': {'severity': '🟡 Medium', 'treatment': 'Improve air circulation, apply sulfur fungicides', 'color': '#8B5CF6'},
    'Smut': {'severity': '🔴 High', 'treatment': 'Use certified seeds, seed treatment', 'color': '#EF4444'},
    'Yellow Rust': {'severity': '🔴 High', 'treatment': 'Apply systemic fungicides early', 'color': '#FBBF24'}
}

# Page config
st.set_page_config(page_title="Wheat Disease AI", page_icon="🌾", layout="wide")

# Compact CSS - fits everything on one screen
st.markdown("""
<style>
    .main {padding: 0.5rem 1rem;}
    .block-container {padding-top: 1rem; padding-bottom: 0rem;}
    h1 {font-size: 2rem; margin: 0; padding: 0;}
    h2 {font-size: 1.3rem; margin-top: 0.5rem;}
    h3 {font-size: 1.1rem; margin: 0.3rem 0;}
    .stMetric {padding: 0.3rem;}
    [data-testid="stMetricValue"] {font-size: 1.5rem;}
    [data-testid="stMetricLabel"] {font-size: 0.85rem;}
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .info-compact {
        background: #F3F4F6;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #10B981;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def predict_disease(image, model):
    img = image.convert('RGB').resize((224, 224))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = (img_array - 0.5) * 2.0
    img_array = np.expand_dims(img_array, 0)
    return model.predict(img_array, verbose=0)[0]

def create_mini_chart(predictions, class_names):
    colors = [DISEASE_INFO[name]['color'] for name in class_names]
    fig = go.Figure(data=[go.Bar(
        x=predictions * 100,
        y=class_names,
        orientation='h',
        marker=dict(color=colors),
        text=[f'{p*100:.0f}%' for p in predictions],
        textposition='inside',
    )])
    fig.update_layout(
        height=250,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(tickfont=dict(size=10)),
        showlegend=False
    )
    return fig

# Main UI
col_header1, col_header2 = st.columns([3, 1])
with col_header1:
    st.markdown("# 🌾 Wheat Disease AI Detector")
with col_header2:
    st.metric("Accuracy", "92.20%")

# Load model
try:
    model = load_model()
except Exception as e:
    st.error(f"Model error: {e}")
    st.stop()

# Upload section
uploaded_file = st.file_uploader("📷 Upload Wheat Leaf Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file:
    image = Image.open(uploaded_file)
    predictions = predict_disease(image, model)
    top_idx = np.argmax(predictions)
    top_class = CLASS_NAMES[top_idx]
    confidence = predictions[top_idx] * 100
    disease_data = DISEASE_INFO[top_class]
    
    # Main content - 2 columns
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.image(image, use_column_width=True)
        
        # Result box
        st.markdown(f"""
        <div class="result-box">
            <h2 style="margin:0; color:white;">{top_class}</h2>
            <h3 style="margin:0.3rem 0; color:white;">{confidence:.1f}% Confidence</h3>
            <p style="margin:0; font-size:1rem;">{disease_data['severity']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Compact metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Primary", top_class[:8]+"..." if len(top_class)>8 else top_class)
        with m2:
            second_idx = np.argsort(predictions)[-2]
            st.metric("Secondary", CLASS_NAMES[second_idx][:8]+"...")
        with m3:
            certainty = "High" if confidence > 80 else "Med" if confidence > 60 else "Low"
            st.metric("Certainty", certainty)
        
        # Mini chart
        fig = create_mini_chart(predictions, CLASS_NAMES)
        st.plotly_chart(fig, use_container_width=True)
    
    # Bottom info - compact
    st.markdown(f"""
    <div class="info-compact">
        <strong>💊 Treatment:</strong> {disease_data['treatment']}
    </div>
    """, unsafe_allow_html=True)

else:
    # No upload state - keep it minimal
    st.info("👆 Upload a wheat leaf image to start AI diagnosis")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Training Images", "18.7K")
    with c2:
        st.metric("Model", "MobileNetV2")
    with c3:
        st.metric("Classes", "6")
    with c4:
        st.metric("Size", "14 MB")
