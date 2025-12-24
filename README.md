# 🏥 Explainable AI (XAI) for Healthcare

A demonstration of trust-based machine learning in clinical contexts. This project uses **SHAP (SHapley Additive exPlanations)** to break down a "black box" Random Forest model predicting breast cancer malignancy.

## 🩺 Objective
In healthcare, knowing *that* a model is 96% accurate is not enough. Clinicians need to know *why* a specific prediction was made. This project implements state-of-the-art interpretability techniques to provide transparency for clinical decision support.

## 🚀 Technical Highlights
- **Engine**: Random Forest Classifier (100 estimators).
- **Interpretability**: SHAP TreeExplainer for global and local insights.
- **Global Strategy**: Beeswarm plots to identify top drivers of malignancy (e.g., worst area, worst concave points).
- **Local Strategy**: Waterfall plots to explain individual patient risk profiles.

## 🛠 Features
- **Dataset**: Breast Cancer Wisconsin (Diagnostic).
- **Performance**: High-precision classification (>95% F1-Score).
- **Automation**: Self-contained script to generate clinical interpretability profiles.

## 💻 Setup
1. **Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Run Analysis**:
   ```bash
   python3 xai_analysis.py
   ```

---
© 2025 Ted Dickey Research
