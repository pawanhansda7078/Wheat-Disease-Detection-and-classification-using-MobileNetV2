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

# Disease information
DISEASE_INFO = {
    'Brown Rust': {
        'severity': 'High',
        'symptoms': 'Orange-brown pustules on leaves',
        'treatment': 'Apply fungicides, remove infected plants',
        'color': '#D97706'
    },
    'Healthy': {
        'severity': 'None',
        'symptoms': 'No visible disease symptoms',
        'treatment': 'Maintain regular care and monitoring',
        'color': '#10B981'
    },
    'Leaf Blight': {
        'severity': 'Medium',
        'symptoms': 'Brown lesions with yellow halos',
        'treatment': 'Use resistant varieties, apply copper-based fungicides',
        'color': '#F59E0B'
    },
    'Mildew': {
        'severity': 'Medium',
        'symptoms': 'White powdery growth on leaves',
        'treatment': 'Improve air circulation, apply sulfur-based fungicides',
        'color': '#8B5CF6'
    },
    'Smut': {
        'severity': 'High',
        'symptoms': 'Black spore masses replace grain',
        'treatment': 'Use certified seeds, apply seed treatment',
        'color': '#EF4444'
    },
    'Yellow Rust': {
        'severity': 'High',
        'symptoms': 'Yellow-orange stripes on leaves',
        'treatment': 'Apply systemic fungicides early',
        'color': '#FBBF24'
    }
}

# Page config
st.set_page_config(
    page_title="Wheat Disease Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS [web:67][web:69][web:71]
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Header styling */
    .header-title {
        font-size: 3rem;
        font-weight: 700;
        color: #1F2937;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #10B981, #059669);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Info cards */
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        border-left: 5px solid #10B981;
    }
    
    /* Disease severity badge */
    .severity-high {
        background: #EF4444;
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .severity-medium {
        background: #F59E0B;
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .severity-none {
        background: #10B981;
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Upload section */
    .uploadedFile {
        border: 2px dashed #10B981;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #10B981, #059669);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(16,185,129,0.4);
    }
</style>
""", unsafe_allow_html=True)

# Load model with caching [web:73]
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Prediction function
def predict_disease(image, model):
    img = image.convert('RGB').resize((224, 224))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = (img_array - 0.5) * 2.0
    img_array = np.expand_dims(img_array, 0)
    predictions = model.predict(img_array, verbose=0)[0]
    return predictions

# Create confidence chart [web:73]
def create_confidence_chart(predictions, class_names):
    colors = [DISEASE_INFO[name]['color'] for name in class_names]
    
    fig = go.Figure(data=[
        go.Bar(
            x=predictions * 100,
            y=class_names,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0.3)', width=1)
            ),
            text=[f'{p*100:.1f}%' for p in predictions],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Confidence Levels for All Classes',
        xaxis_title='Confidence (%)',
        yaxis_title='Disease Class',
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        showlegend=False
    )
    
    return fig

# Main app
def main():
    # Header
    st.markdown('<h1 class="header-title">🌾 Wheat Disease Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="header-subtitle">AI-Powered Disease Diagnosis using MobileNetV2 | Accuracy: 92.20%</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/wheat.png", width=80)
        st.title("About")
        st.info("""
        This application uses deep learning to detect wheat diseases from leaf images.
        
        **Model:** MobileNetV2  
        **Training:** 18,752 images  
        **Validation Accuracy:** 92.20%  
        **Classes:** 6 diseases
        """)
        
        st.markdown("---")
        st.subheader("📋 Instructions")
        st.markdown("""
        1. Upload a clear image of wheat leaf
        2. Ensure good lighting
        3. Focus on disease symptoms
        4. Wait for AI analysis
        """)
        
        st.markdown("---")
        st.caption("Developed for Agricultural Research")
    
    # Load model
    try:
        with st.spinner("Loading AI model..."):
            model = load_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return
    
    # Main content
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📷 Upload Wheat Leaf Image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of wheat leaf for disease detection"
    )
    
    if uploaded_file:
        # Display image and results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True, caption="Original Image")
            
            # Image info
            st.caption(f"Size: {image.size[0]}x{image.size[1]} | Format: {image.format}")
        
        with col2:
            st.subheader("🔬 AI Analysis")
            
            with st.spinner("Analyzing image..."):
                predictions = predict_disease(image, model)
                top_idx = np.argmax(predictions)
                top_class = CLASS_NAMES[top_idx]
                confidence = predictions[top_idx] * 100
                disease_data = DISEASE_INFO[top_class]
            
            # Result card
            st.markdown(f"""
            <div class="result-card">
                <h2 style="color: {disease_data['color']}; margin-bottom: 1rem;">
                    {top_class}
                </h2>
                <p style="font-size: 1.5rem; font-weight: 600; color: #1F2937;">
                    Confidence: {confidence:.2f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Severity badge
            severity_class = f"severity-{disease_data['severity'].lower()}"
            st.markdown(f'<span class="{severity_class}">Severity: {disease_data["severity"]}</span>', 
                       unsafe_allow_html=True)
        
        # Detailed information
        st.markdown("---")
        st.subheader("📊 Detailed Analysis")
        
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.metric(
                label="Primary Diagnosis",
                value=top_class,
                delta=f"{confidence:.1f}% confidence"
            )
        
        with col4:
            st.metric(
                label="Secondary Prediction",
                value=CLASS_NAMES[np.argsort(predictions)[-2]],
                delta=f"{predictions[np.argsort(predictions)[-2]]*100:.1f}%"
            )
        
        with col5:
            st.metric(
                label="Model Certainty",
                value="High" if confidence > 80 else "Medium" if confidence > 60 else "Low",
                delta=f"{confidence:.0f}%"
            )
        
        # Disease information
        st.markdown("---")
        col6, col7 = st.columns(2)
        
        with col6:
            st.subheader("🦠 Symptoms")
            st.write(disease_data['symptoms'])
            
        with col7:
            st.subheader("💊 Recommended Treatment")
            st.write(disease_data['treatment'])
        
        # Confidence chart
        st.markdown("---")
        st.subheader("📈 All Class Probabilities")
        fig = create_confidence_chart(predictions, CLASS_NAMES)
        st.plotly_chart(fig, use_container_width=True)
        
        # Download results
        st.markdown("---")
        if st.button("📥 Download Analysis Report"):
            report = f"""
            WHEAT DISEASE DETECTION REPORT
            ===============================
            
            Primary Diagnosis: {top_class}
            Confidence: {confidence:.2f}%
            Severity: {disease_data['severity']}
            
            Symptoms: {disease_data['symptoms']}
            Treatment: {disease_data['treatment']}
            
            All Probabilities:
            {chr(10).join([f"  - {CLASS_NAMES[i]}: {predictions[i]*100:.2f}%" for i in range(len(CLASS_NAMES))])}
            """
            st.download_button(
                label="Download Report",
                data=report,
                file_name=f"wheat_disease_report_{top_class.replace(' ', '_')}.txt",
                mime="text/plain"
            )
    
    else:
        # Welcome section
        st.info("👆 Upload a wheat leaf image to begin analysis")
        
        # Sample statistics
        st.markdown("---")
        st.subheader("📊 Model Performance")
        
        col8, col9, col10, col11 = st.columns(4)
        
        with col8:
            st.metric("Training Images", "18,752")
        with col9:
            st.metric("Validation Accuracy", "92.20%")
        with col10:
            st.metric("Disease Classes", "6")
        with col11:
            st.metric("Model Size", "~14 MB")

if __name__ == "__main__":
    main()
