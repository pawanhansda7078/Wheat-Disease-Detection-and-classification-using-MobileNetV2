import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Set environment variable at the very top
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# --- CONFIGURATION ---
MODEL_PATH = "wheat_model_final.h5"
IMAGE_SIZE = (224, 224)

# Class names in correct order
CLASS_NAMES = [
    'Brown Rust',
    'Healthy',
    'Leaf Blight',
    'Mildew',
    'Smut',
    'Yellow Rust'
]


@st.cache_resource
def load_model():
    """Loads the trained Keras model."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        st.error(f"Looking in: {os.getcwd()}")
        st.stop()
    
    try:
        # Load with compile=False
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        
        # Recompile
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        import traceback
        st.error(traceback.format_exc())
        st.stop()


def preprocess_image(img_file):
    """Preprocess uploaded image using PIL"""
    img = Image.open(img_file)
    img = img.convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img)
    
    # Normalize to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    # MobileNetV2 preprocessing: scale to [-1, 1]
    img_array = (img_array - 0.5) * 2.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def preprocess_and_predict(model, uploaded_file):
    """Preprocesses the uploaded image and returns predictions."""
    try:
        # Preprocess
        img_array = preprocess_image(uploaded_file)

        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        
        # Get top 2 predictions
        top_2_indices = np.argsort(predictions[0])[-2:][::-1]
        top_2_classes = [CLASS_NAMES[idx] for idx in top_2_indices]
        top_2_confidences = [round(100 * predictions[0][idx], 2) for idx in top_2_indices]
        
        return top_2_classes, top_2_confidences, predictions[0]
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None


def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Wheat Disease Detection", page_icon="🌾")
    
    st.title("🌾 Wheat Disease Detection System")
    st.write("Upload a wheat leaf image to detect diseases using MobileNetV2 deep learning model.")
    st.write("**Model Accuracy:** 92.20% on validation set")

    # Load model
    model = load_model()
    st.success("✅ Model loaded successfully!")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a wheat leaf image...", 
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )

    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.write("### Processing...")
            
            # Get predictions
            top_2_classes, top_2_confidences, all_predictions = preprocess_and_predict(
                model, uploaded_file
            )

            if top_2_classes and top_2_confidences:
                # Display top prediction
                st.markdown("---")
                st.subheader("🎯 Prediction Results")
                
                # Top prediction with confidence indicator
                if top_2_confidences[0] > 85:
                    confidence_color = "green"
                elif top_2_confidences[0] > 70:
                    confidence_color = "orange"
                else:
                    confidence_color = "red"
                
                st.markdown(f"""
                ### **Primary Diagnosis:** {top_2_classes[0]}
                **Confidence:** :{confidence_color}[{top_2_confidences[0]}%]
                """)
                
                # Second prediction
                st.markdown(f"""
                **Alternative:** {top_2_classes[1]} ({top_2_confidences[1]}%)
                """)
                
                # Confidence interpretation
                if top_2_confidences[0] > 90:
                    st.success("🟢 Very High Confidence - Model is very certain about this diagnosis.")
                elif top_2_confidences[0] > 80:
                    st.success("🟢 High Confidence - Diagnosis is reliable.")
                elif top_2_confidences[0] > 70:
                    st.warning("🟡 Moderate Confidence - Consider consulting an expert.")
                else:
                    st.warning("🟠 Low Confidence - Please upload a clearer image or consult an expert.")
                
                # Show all class probabilities
                with st.expander("📊 View All Class Probabilities"):
                    for i, class_name in enumerate(CLASS_NAMES):
                        prob = round(100 * all_predictions[i], 2)
                        st.write(f"**{class_name}:** {prob}%")
                        st.progress(float(all_predictions[i]))
    else:
        st.info("👆 Please upload a wheat leaf image to begin detection.")
        
        # Show example information
        with st.expander("ℹ️ About This Model"):
            st.write("""
            **Model Details:**
            - Architecture: MobileNetV2 (Transfer Learning)
            - Training Dataset: 18,752 wheat leaf images
            - Validation Accuracy: 92.20%
            - Classes Detected: 6 wheat diseases/conditions
            
            **Per-Class Performance:**
            1. Brown Rust (91.06% recall)
            2. Healthy (93.84% recall)
            3. Leaf Blight (86.24% recall)
            4. Mildew (99.59% recall) 🏆
            5. Smut (100% recall) 🏆
            6. Yellow Rust (87.05% recall)
            
            **Tips for Best Results:**
            - Use clear, well-lit images
            - Focus on the affected leaf area
            - Avoid blurry or low-resolution images
            """)


if __name__ == "__main__":
    main()
