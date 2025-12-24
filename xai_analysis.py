import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')

def run_healthcare_xai():
    print("--- [1/4] Loading Healthcare Dataset (Breast Cancer Wisconsin) ---")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n--- [2/4] Training Interpreted Model (Random Forest) ---")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Quick check on performance
    y_pred = model.predict(X_test)
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    
    print("\n--- [3/4] Initializing SHAP Explainer (Explaining Clinical Predictions) ---")
    explainer = shap.TreeExplainer(model)
    # Get explanation object which is more modern
    explanation = explainer(X_test)
    
    print("\n--- [4/4] Generating Interpretability Visualizations ---")
    
    # For binary classification, explanation has shape (N, D, 2)
    # We want class 0 (Malignant) to explain risk factors
    # or class 1 (Benign). Usually index 0 in this dataset is Malignant.
    
    plt.figure(figsize=(12, 8))
    # explanation[:, :, 0] explains the 'malignant' class
    shap.plots.beeswarm(explanation[:, :, 0], show=False)
    plt.title("SHAP Summary: Drivers of Malignant Classification", fontsize=14)
    plt.tight_layout()
    plt.savefig('healthcare_xai_summary.png', dpi=300)
    print("Summary plot saved: healthcare_xai_summary.png")
    
    # Individual explanation
    patient_idx = 10
    plt.figure(figsize=(12, 4))
    shap.plots.waterfall(explanation[patient_idx, :, 0], show=False)
    plt.title(f"Patient {patient_idx} Interpretability Profile", fontsize=14)
    plt.tight_layout()
    plt.savefig('patient_explanation.png', dpi=300)
    print("Patient-level explanation saved: patient_explanation.png")

if __name__ == "__main__":
    run_healthcare_xai()
