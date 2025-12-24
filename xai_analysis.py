import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

# 1. Load Dataset (Wisconsin Diagnostic)
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Explainability (SHAP)
# Using Explainer for the model
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# For binary classification, slice for class 1 (Malignant in this context, or typically the target of interest)
if len(shap_values.values.shape) == 3:
    shap_val_output = shap_values[:, :, 1]
else:
    shap_val_output = shap_values

# 5. Visualizations - ACADEMIC LIGHT THEME
plt.style.use('seaborn-v0_8-muted') # Close to standard academic styles
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'font.family': 'sans-serif',
    'font.size': 12
})

# Create a composite figure (Tall and Slim for 'scroll' effect)
fig = plt.figure(figsize=(10, 20), constrained_layout=True)
gs = fig.add_gridspec(2, 1)

# Panel 1: Global Summary (Beeswarm)
ax1 = fig.add_subplot(gs[0])
plt.sca(ax1)
shap.plots.beeswarm(shap_val_output, show=False, color_bar=True)
plt.title("Global Clinical Drivers (Beeswarm Analysis)", fontsize=20, pad=40, fontweight='bold', color='#1a1a1a')

# Panel 2: Local Patient Explanation (Waterfall)
ax2 = fig.add_subplot(gs[1])
plt.sca(ax2)
patient_idx = 10 
shap.plots.waterfall(shap_val_output[patient_idx], show=False)
plt.title(f"Patient ID: 10 - Diagnostic Attribution (Waterfall)", fontsize=20, pad=40, fontweight='bold', color='#1a1a1a')

# Save the full composite image
plt.savefig("/Volumes/Samsung 990 Pro/dev/healthcare-xai/healthcare_xai_full_scroll.png", dpi=300, facecolor='white', bbox_inches='tight')

print("✅ Visualization composite generated (Academic Light Theme): healthcare_xai_full_scroll.png")
