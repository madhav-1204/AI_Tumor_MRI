# 🧠 AI Brain Tumor Classification System

An explainable, agentic AI system for brain tumor classification using deep learning and large language models.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)

## ⚠️ IMPORTANT DISCLAIMER

**This is a demonstration project for educational purposes only. NOT for medical use.**

This AI system is not FDA-approved and should never be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical advice.

## 🎯 Features

- **Multi-Agent Architecture**: Orchestrated system with specialized agents
  - 👁️ **Vision Agent**: EfficientNet-B0 based classifier
  - 🎨 **Explainability Agent**: Grad-CAM visualization
  - 🧠 **Reasoning Agent**: LLM-powered explanations  
  - 📄 **Report Agent**: Structured diagnostic reports

- **4-Class Classification**:
  - Glioma
  - Meningioma
  - Pituitary Tumor
  - No Tumor

- **Explainable AI**: Grad-CAM heatmaps showing decision factors
- **Interactive Web Interface**: Built with Streamlit
- **Professional Reports**: Clinical-style diagnostic reports

## 🚀 Quick Start

### Prerequisites

- Python 3.11+ (recommended) or 3.10+
- Anaconda/Miniconda (recommended) OR Python standalone
- 4GB+ RAM
- Internet connection (for model download)

### Installation

#### Step 1: Clone the Repository

```bash
git clone https://github.com/spargy4050v/AI_Tumor.git
cd AI_Tumor
```

#### Step 2: Create Virtual Environment

**Option A: Using Conda (Recommended):**
```bash
# Create environment with Python 3.11
conda create -n ai_tumor_311 python=3.11 -y

# Activate environment
conda activate ai_tumor_311
```

**Option B: Using Python venv:**
```bash
# Create virtual environment with Python 3.11
# Windows (if you have Python 3.11 installed):
py -3.11 -m venv venv

# Linux/Mac (if you have Python 3.11 installed):
python3.11 -m venv venv

# Or use default Python (make sure it's 3.11+):
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** Installation downloads ~150MB of packages including PyTorch, Streamlit, and ML libraries. This may take 2-5 minutes depending on your internet speed.

This will install:
- PyTorch & torchvision
- timm (PyTorch Image Models)
- Streamlit
- grad-cam (Gradient-weighted Class Activation Mapping)
- google-generativeai (optional, for LLM explanations)
- matplotlib, opencv-python, pillow, numpy, pandas

#### Step 4: (Optional) Set Up Gemini API

For enhanced LLM-generated explanations:

1. Get a free API key: https://makersuite.google.com/app/apikey
2. Create a `.env` file:
```bash
cp .env.example .env
```
3. Edit `.env` and add your key:
```
GEMINI_API_KEY=your_actual_api_key_here
```

**Note**: The system works without an API key using fallback explanations.

### Running the Application

#### Web Interface (Recommended)

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

#### Command Line Testing

```python
from agents.orchestrator import Orchestrator

# Initialize
orchestrator = Orchestrator()
orchestrator.load_model()

# Process image
results = orchestrator.process_image('path/to/mri_image.jpg')

# View results
print(results['report'])
```

## 📁 Project Structure

```
AI_Tumor/
├── agents/
│   ├── __init__.py
│   ├── vision_agent.py          # Image classification
│   ├── explainability_agent.py  # Grad-CAM visualization
│   ├── reasoning_agent.py       # LLM explanations
│   ├── report_agent.py          # Report generation
│   └── orchestrator.py          # Agent coordination
├── app.py                       # Streamlit web interface
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
└── README.md
```

## 🔬 How It Works

### Agent-Based Architecture

```
User Input (MRI Image)
         ↓
   [Orchestrator]
         ↓
    ┌────┴────┬──────────┬───────────┐
    ↓         ↓          ↓           ↓
[Vision] [Explain] [Reasoning] [Report]
    ↓         ↓          ↓           ↓
    └────┬────┴──────────┴───────────┘
         ↓
   Final Results
```

1. **Vision Agent**: Runs EfficientNet-B0 inference
2. **Explainability Agent**: Generates Grad-CAM heatmaps
3. **Reasoning Agent**: Creates human-readable explanations
4. **Report Agent**: Compiles structured diagnostic report
5. **Orchestrator**: Coordinates all agents

## 📊 Usage

### Web Interface

1. Launch: `streamlit run app.py`
2. Click "Initialize System"
3. Upload MRI image (JPG/PNG)
4. Click "Analyze Image"
5. View results, heatmaps, and reports

### Python API

```python
from agents.orchestrator import Orchestrator

orchestrator = Orchestrator(gemini_api_key="optional_key")
orchestrator.load_model()

results = orchestrator.process_image('brain_mri.jpg')

print(f"Prediction: {results['prediction']['predicted_class']}")
print(f"Confidence: {results['prediction']['confidence']:.2f}%")
```

## 🧪 Testing

```bash
# Test vision agent
python agents/vision_agent.py

# Test orchestrator
python agents/orchestrator.py
```

## 🚢 Deployment

### Streamlit Cloud
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository and deploy

### Hugging Face Spaces
1. Create Space with Streamlit SDK
2. Push code
3. Deploy

## 🛠️ Troubleshooting

**Model Download Issues**: Check internet connection, model downloads once (~20MB)

**Memory Errors**: Close other apps, system uses CPU automatically

**Grad-CAM Not Showing**: Ensure `grad-cam` package installed

**API Key Issues**: System works without key in fallback mode

## ⚖️ Legal Disclaimer

This software is provided "as is" for educational purposes only.

**NOT FOR CLINICAL USE.**

- Not validated for medical diagnosis
- Not FDA approved
- No warranty or guarantee
- Developers assume no liability
- Always consult medical professionals

---

**Built for AI Hackathon 2025** | GitHub: [@spargy4050v](https://github.com/spargy4050v)
