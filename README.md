# Intelligent Infant Health Predictor With Personalized AI Guidance

> **Mission**: Leveraging Advanced Machine Learning and Generative AI to identify mortality risks in infants and provide actionable, personalized survival plans.

![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.9+-blue.svg) ![Status](https://img.shields.io/badge/status-research_demo-orange.svg)

## 📖 Project Overview

This application serves as a **Clinical Decision Support System (CDSS)** demo. It predicts the risk of infant mortality based on key health indicators (birth weight, maternal age, immunization status, etc.) and uses Generative AI to create personalized care plans ("Survival Plans").

The system is designed to be **transparent** (via SHAP explanations) and **actionable** (via LLM-generated guidance), helping healthcare workers prioritize high-risk cases.

> [!WARNING]
> **Disclaimer**: This project uses synthetic data for demonstration purposes. **Do NOT use for real clinical diagnosis.** Replace the model with one trained on validated medical datasets (e.g., DHS, WHO) before real-world deployment.

---

## ✨ Key Features

### 🔍 Precision Risk Prediction
- **Model Selection**: We conducted extensive performance testing across multiple algorithms (Random Forest, XGBoost, Logistic Regression). **CatBoost** consistently delivered the highest accuracy and stability.
- **Ensemble Strategy**: While a Voting Classifier is available, **CatBoost** is leveraged as the primary high-performance engine for risk prediction.
- **Robustness**: Validated using 10-fold Cross-Validation (CV=10) to ensure reliability against data leakage.

### 🧠 Explainable AI (XAI)
- **SHAP Integration**: Uses **SHAP (Shapley Additive Explanations)** to explain *why* a specific prediction was made.
- **Visual Insights**: Shows which factors (e.g., "Low Birth Weight", "Poor Nutrition") increased or decreased the risk for each specific patient.

### 🤖 Generative AI Guidance
- **Personalized Care Plans**: Integrates with **Hugging Face Inference API** (e.g., Zephyr/Mistral models) to generate detailed, WHO-aligned survival plans for years 0-5.
- **Context-Aware**: The AI considers the specific risk factors of the child (e.g., "Since the child has low birth weight, emphasize thermal care...") when generating advice.

### ⚖️ Data Fairness & Handling
- **SMOTE Balancing**: Applies **Synthetic Minority Over-sampling Technique (SMOTE)** ONLY on training data to address class imbalance without causing data leakage.
- **Pipeline Architecture**: Uses `imblearn` pipelines to ensure correct data processing during cross-validation.

### 🛡️ Secure History & Dashboard
- **Patient History**: SQLite database stores all predictions, explanations, and generated plans.
- **Session Auth**: Secure login system to protect patient data.
- **Modern UI**: Responsive, dark-mode capable interface built with dynamic CSS.

---

## 🏗️ Technical Architecture

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Backend** | Flask (Python) | REST API handling predictions, history, and LLM orchestration. |
| **ML Engine** | Scikit-Learn, CatBoost | Voting Ensemble model with SMOTE balancing. |
| **Explainability** | SHAP | Generates local feature attribution values. |
| **GenAI** | Hugging Face API | Generates natural language care plans. |
| **Database** | SQLite | Lightweight persistent storage for history and users. |
| **Frontend** | HTML/JS + Tailwind | Interactive dashboard (served statically by Flask). |

---

## 📂 Directory Structure

```plaintext
Child-Mortality-Prediction/
├── app.py                 # 🚀 Main Flask Entrypoint (API + Server)
├── train_model.py         # 🧠 Model Training Script (Ensemble + SMOTE)
├── requirements.txt       # 📦 Python Dependencies
├── history.db             # 💾 SQLite Database (Auto-created)
├── models/
│   ├── model.pkl          # Trained VotingClassifier pipeline
│   ├── features.pkl       # Feature names for consistency
│   └── sample_input.csv   # Sample data for heuristics
├── static/
│   ├── index.html         # Frontend Dashboard
│   ├── app.js             # Frontend Logic (API calls, UI rendering)
│   └── styles.css         # Styling
└── README.md              # Project Documentation
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/child-mortality-prediction.git
cd child-mortality-prediction
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configuration (.env)
Create a `.env` file in the root directory to enable the Generative AI features:
```ini
HF_API_KEY=your_hugging_face_token_here
HF_MODEL=HuggingFaceH4/zephyr-7b-beta
FLASK_SECRET_KEY=super-secret-key-change-me
```
*Note: You can get a free token from [Hugging Face](https://huggingface.co/settings/tokens).*

---

## 🏃 Usage Guide

### Step 1: Train the Model
Before running the app, you need to generate the model. This script offers to generate synthetic data if none exists.
```bash
python train_model.py
```
*This will train the Voting Classifier using SMOTE and save it to `models/model.pkl`.*

### Step 2: Run the Application
```bash
python app.py
```
Output:
```
🚀 Starting Child Mortality API server on http://127.0.0.1:8000
```

### Step 3: Access Dashboard
Open your browser and navigate to:
**http://127.0.0.1:8000**

1.  **Register/Login**: Create a local account to save history.
2.  **Enter Data**: Fill in the child's health metrics (Weight, Age, Nutrition, etc.).
3.  **Predict**: Click "Analyze Risk".
4.  **View Results**:
    -   See the **Risk Probability**.
    -   Read the **AI Survival Plan**.
    -   Expand "Explain Prediction" to see the **SHAP Analysis**.

---

## 🤝 Contribution
Contributions are welcome! Please fork the repository and submit a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request
