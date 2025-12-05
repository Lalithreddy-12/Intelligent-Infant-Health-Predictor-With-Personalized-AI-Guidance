# Intelligent Infant Health Predictor With Personalized AI Guidance (Backend + Frontend)

This is a minimal demo application inspired by the project paper you provided. It includes a Flask backend exposing a `/api/predict` endpoint and a static frontend (`static/index.html`) that consumes it.

**Important:** This demo uses a synthetically generated dataset for illustration only. Do **NOT** use this model for real-world decisions or healthcare purposes. Replace the synthetic training data with validated datasets and rigorously validate model fairness, privacy, and clinical safety before any real deployment.

## Files
- `train_model.py` - generates synthetic data, trains a RandomForest model, and saves `models/model.pkl`
- `models/model.pkl` - created by running the train script
- `app.py` - Flask app serving the frontend and prediction API
- `static/index.html`, `static/app.js` - frontend files
- `requirements.txt` - Python dependencies

## How to run (locally)
1. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # Windows: venv\\Scripts\\activate
   pip install -r requirements.txt
   ```
2. Train the model (creates `models/model.pkl`):
   ```bash
   python train_model.py
   ```
3. Start the Flask app:
   ```bash
   python app.py
   ```
4. Open your browser at `http://127.0.0.1:8000` and try the demo.

## Notes and next steps
- Replace synthetic data with a real dataset (e.g., DHS, WHO) after ensuring permissions and privacy compliance.
- Add explainability (SHAP/LIME), bias checks, logging, tests, and CI/CD for production readiness.
