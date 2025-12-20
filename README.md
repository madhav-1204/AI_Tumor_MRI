# NeuroVision AI - Agentic Brain Tumor Analysis System

**NeuroVision AI** is an advanced, agentic AI system designed to assist neuroradiologists in classifying brain tumors from MRI scans. It leverages a multi-agent architecture to perceive, reason, and report findings with clinical precision.

## 🧠 Key Features

*   **Multi-Agent Architecture**:
    *   **Vision Agent**: Deep learning (ResNet/EfficientNet) for tumor classification (Glioma, Meningioma, Pituitary, No Tumor).
    *   **Explainability Agent**: Generates Grad-CAM heatmaps to visualize the model's focus.
    *   **Reasoning Agent**: Uses **Google Gemini 1.5 Flash** to provide detailed clinical interpretation and morphological analysis.
    *   **Orchestrator**: Coordinates agents and manages complex workflows (Single vs. Multi-View).
*   **Conflict Detection**: Automatically detects diagnostic inconsistencies across multiple views of the same patient and suppresses AI interpretation to ensure safety.
*   **Professional Reporting**: Auto-generates comprehensive, clinically formatted PDF reports with embedded images and analysis key findings.
*   **Privacy-First Design**: Local processing for visuals; streamlined UI with no unnecessary technical jargon.

---

## 🛠️ Installation & Setup

You can set up NeuroVision AI using either **Conda** or standard **Python (venv)**. Ensure you have **Python 3.11** installed.

### Option A: Using Conda (Recommended)

1.  **Create the environment**:
    ```bash
    conda create -n ai_tumor python=3.11
    ```
2.  **Activate the environment**:
    ```bash
    conda activate ai_tumor
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Option B: Using Python venv (Standard)

1.  **Create a virtual environment**:
    ```bash
    # Windows
    python -m venv venv
    
    # macOS/Linux
    python3 -m venv venv
    ```
2.  **Activate the environment**:
    ```bash
    # Windows
    .\venv\Scripts\activate
    
    # macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## ⚙️ Configuration

1.  **Gemini API Key**: This system requires a Google Gemini API key for the Reasoning Agent.
    *   Create a file named `.env` in the root directory.
    *   Add your key:
        ```env
        GEMINI_API_KEY=your_actual_api_key_here
        ```

2.  **Streamlit Config**:
    *   A `.streamlit/config.toml` file is included to ensure compatibility with PyTorch. Do not remove it.

---

## 🚀 Running the Application

To start the web interface, run:

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## 📂 Project Structure

```
AI_Tumor/
├── agents/             # Core AI Agents
│   ├── vision_agent.py
│   ├── reasoning_agent.py
│   ├── report_agent.py
│   └── orchestrator.py
├── utils/              # Utilities (PDF Generation)
├── models/             # PyTorch Model weights
├── .streamlit/         # Streamlit Configuration
├── app.py              # Main Streamlit Interface
├── requirements.txt    # Project Dependencies
└── README.md           # Documentation
```

## ⚠️ Medical Disclaimer

This tool is for **research and educational purposes only**. It is not a certified medical device and should not be used for primary diagnosis. All findings must be verified by a qualified radiologist.
