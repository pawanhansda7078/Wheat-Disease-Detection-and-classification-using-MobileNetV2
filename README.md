# 🌾 Wheat Disease Detection and Classification using MobileNetV2

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red)
![Accuracy](https://img.shields.io/badge/Accuracy-92.20%25-green)

AI-powered wheat disease detection system using deep learning with MobileNetV2 architecture. This system can accurately identify 6 different wheat diseases from leaf images with 92.20% validation accuracy.

## 📊 Model Performance

- **Architecture:** MobileNetV2 with custom classification head
- **Training Images:** 18,752
- **Validation Accuracy:** 92.20%
- **Model Size:** ~14 MB
- **Inference Time:** < 1 second per image

## 🎯 Disease Classes

The model can detect the following wheat diseases:

1. **Brown Rust** - High severity fungal disease
2. **Healthy** - No disease symptoms
3. **Leaf Blight** - Medium severity bacterial infection
4. **Mildew** - Medium severity powdery fungal growth
5. **Smut** - High severity fungal disease affecting grain
6. **Yellow Rust** - High severity stripe rust disease

## 🚀 Quick Start

### Prerequisites

- Python 3.10
- Windows OS (for .bat launcher)
- 4GB RAM minimum
- GPU optional (for faster inference)

### Installation

1. **Clone the repository**
bash
git clone https://github.com/pawanhansda7078/Wheat-Disease-Detection-and-classification-using-MobileNetV2.git
cd Wheat-Disease-Detection-and-classification-using-MobileNetV2

2. Create virtual environment
bash
python -m venv wheat_env

3. Activate virtual environment
Windows:
bash
wheat_env\Scripts\activate

Linux/Mac:
bash
source wheat_env/bin/activate

4. Install dependencies
bash
pip install -r requirements.txt


Running the Application
Option 1: One-Click Launcher (Windows)
bash
# Simply double-click
run_wheat_app.bat

Option 2: Manual Launch
bash
# Activate environment first
wheat_env\Scripts\activate

# Run compact version (recommended)
streamlit run app_wheat_compact.py

# Or run enhanced version
streamlit run app_wheat_enhanced.py
The application will open in your default web browser at http://localhost:8501

📁 Project Structure
text
Wheat-Disease-Detection/
├── wheat_model_finetuned.h5      # Main trained model (recommended)
├── wheat_model_best.h5            # Best checkpoint during training
├── wheat_model_final.h5           # Final epoch model
├── app_wheat_compact.py           # Compact UI (single page, no scroll)
├── app_wheat_enhanced.py          # Enhanced UI with detailed analysis
├── app_wheat_simple.py            # Simple baseline version
├── train_model.py                 # Training script for CPU
├── train_model_gpu.py             # Training script optimized for GPU
├── evaluate_model.py              # Model evaluation utilities
├── run_wheat_app.bat              # Windows launcher script
├── requirements.txt               # Python dependencies
├── document (1).pdf               # Project documentation
├── .gitignore                     # Git ignore file
└── README.md                      # This file

🎨 User Interface Features
Compact Version (app_wheat_compact.py)
✅ Single-page layout (no scrolling)
✅ Quick disease prediction
✅ Confidence scores visualization
✅ Treatment recommendations
✅ Lightweight and fast

Enhanced Version (app_wheat_enhanced.py)
✅ Detailed disease information
✅ Interactive Plotly charts
✅ Symptoms and severity indicators
✅ Comprehensive analysis
✅ Download report feature
✅ Model performance metrics

🧠 Model Architecture
python
MobileNetV2 (ImageNet pretrained)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization → Dense(512, relu, L2=0.01) → Dropout(0.5)
    ↓
BatchNormalization → Dense(256, relu, L2=0.01) → Dropout(0.4)
    ↓
Dense(6, softmax) [Output Layer]

#Training Configuration
Base Model: MobileNetV2 (ImageNet weights)

Fine-tuning: Last 50 layers unfrozen

Optimizer: Adam

Learning Rate: 0.0001

Batch Size: 32

Image Size: 224×224×3

Data Augmentation: Rotation, flip, zoom, brightness

Regularization: L2 (0.01), Dropout (0.5, 0.4)

📈 Training the Model
To retrain the model with your own dataset:
bash
# Prepare your dataset in this structure:
dataset/
├── train/
│   ├── Brown Rust/
│   ├── Healthy/
│   ├── Leaf Blight/
│   ├── Mildew/
│   ├── Smut/
│   └── Yellow Rust/
└── validation/
    ├── Brown Rust/
    ├── Healthy/
    └── ...

# Run training
python train_model.py

# For GPU training
python train_model_gpu.py

🔬 Evaluation
Evaluate model performance:
bash
python evaluate_model.py
This will generate:
Confusion matrix
Classification report
Per-class accuracy
ROC curves

📦 Dependencies
tensorflow==2.13.0
streamlit==1.28.0
pillow==10.0.0
numpy==1.24.3
plotly==5.18.0
protobuf==4.23.4

🛠️ Troubleshooting
Issue: Protobuf Error
bash
pip uninstall protobuf -y
pip install protobuf==4.23.4
Issue: Model Not Loading
Ensure wheat_model_finetuned.h5 is in the same directory

Check if virtual environment is activated

Verify TensorFlow version: python -c "import tensorflow; print(tensorflow.__version__)"

Issue: Streamlit Won't Start
bash
# Clear Streamlit cache
streamlit cache clear

# Restart with
streamlit run app_wheat_compact.py

🎓 Academic Information
Project: BTech Final Year Major Project
Institution: AKTU (Dr. A.P.J. Abdul Kalam Technical University)
Department: Computer Science and Engineering
Year: 2025-2026

📊 Results
Metric	Value
Training Accuracy	95.8%
Validation Accuracy	92.20%
Test Accuracy	91.5%
Average Precision	0.924
Average Recall	0.918
F1-Score	0.921

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

📝 License
This project is for academic purposes.

👨‍💻 Author
Pawan Hansda
GitHub: @pawanhansda7078
Email: hansdapawan2811@gmail.com

🙏 Acknowledgments
Dataset sourced from agricultural research databases
MobileNetV2 architecture by Google Research
AKTU for academic support
TensorFlow and Streamlit communities

📚 References
Sandler, M., et al. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
Transfer Learning for Computer Vision (TensorFlow Documentation)
Wheat Disease Detection using Deep Learning - Research Papers

🔗 Links
Live Demo (if deployed)
Documentation
Report Issues

⭐ Star this repository if you find it helpful!

Made with ❤️ for agricultural technology and AI research

