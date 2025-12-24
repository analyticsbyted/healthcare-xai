# 🏥 Explainable AI (XAI) for Clinical Decision Support: A Game-Theoretic Approach to Oncological Diagnostics

## 🗒 Abstract
In high-stakes clinical environments, the "Black Box" nature of advanced machine learning models poses significant barriers to adoption. This research project demonstrates the application of **SHapley Additive exPlanations (SHAP)**—a game-theoretic framework—to interpret an ensemble-based diagnostic model. We achieve a balance between high-fidelity predictive power (96% accuracy) and human-centric transparency, providing clinicians with actionable insights into the underlying drivers of tumor malignancy.

## 🩺 The Clinical Interpretability Gap
Modern oncology increasingly relies on computational methods for early diagnosis. However, traditional metrics like accuracy and AUC-ROC fail to provide the *justification* required for clinical interventions. This project addresses the "Trust Gap" by implementing an **Explainable AI (XAI)** layer that bridges the distance between complex feature spaces and bedside decision-making.

## 🔬 Methodology & Implementation

### 1. Data Source & Preprocessing
The analysis utilizes the **Breast Cancer Wisconsin (Diagnostic) Dataset** ($N=569$). Features were engineered from digitized images of fine needle aspirates (FNA), capturing characteristics such as radius, texture, perimeter, and area.

### 2. Predictive Modeling (The Ensemble Layer)
A **Random Forest Classifier** ($k=100$ estimators) was implemented as the diagnostic engine. Ensemble methods were selected for their ability to capture non-linear feature interactions while maintaining robust generalization.

### 3. Interpretability Framework (Game Theory Integration)
We leverage **TreeSHAP**, an additive feature attribution method. By treating feature contributions as participants in a cooperative game, we calculate the **Shapley values**, ensuring a fair distribution of the prediction's payout among the features. This allows for:
- **Global Explanations**: Identification of population-level diagnostic drivers.
- **Local Explanations**: Patient-specific risk profiles for individualized care.

## 📊 Empirical Results & Performance Evaluation

The model achieved high-precision results across all diagnostic tiers:

| Metric        | Malignant (Class 0) | Benign (Class 1) | Macro Average |
|---------------|---------------------|------------------|---------------|
| **Precision** | 0.98                | 0.96             | 0.97          |
| **Recall**    | 0.93                | 0.99             | 0.96          |
| **F1-Score**  | 0.95                | 0.97             | 0.96          |
| **Accuracy**  | --                  | --               | **0.96**      |

## 🧠 Hierarchical Interpretation

### Global Drivers (Beeswarm Analysis)
Our global interpretation reveals that **'Worst Area'** and **'Worst Concave Points'** are the primary predictors of malignancy. High values in these dimensions consistently increase the SHAP value, pushing the model's prediction toward a malignant classification.

### Local Clinical Profiles (Waterfall Analysis)
The local interpretability layer (see `patient_explanation.png`) allows clinicians to view the positive and negative "pushes" for a single patient. In our test case (Patient 10), the model correctly identified a high-risk scenario driven primarily by **'Worst Perimeter'** and **'Area Error'**, even when other features were within normal bounds.

## 🚀 Implications for Healthcare
The integration of SHAP values into clinical workflows allows for:
1. **Error Auditing**: Identifying when a model is relying on "spurious" correlations.
2. **Clinical Validation**: Ensuring algorithmic decisions align with established medical knowledge (e.g., cell morphology).
3. **Patient Communication**: Providing visual evidence to patients regarding their diagnostic classification.

---
© 2025 Ted Dickey Research | Analytics for Clinical Trust (ACT)

## 📊 R-Based Supplemental Analysis
A supplemental report built in **R-Markdown** is available in this repository: `xai_report.Rmd`.
- **Framework**: DALEX (Descriptive mAchine Learning Explanations).
- **Objective**: Demonstrates cross-platform verification of clinical feature importance.
- **Usage**: Open `xai_report.Rmd` in RStudio and click "Knit" to generate the full HTML research report. This report is optimized for publication on **RPubs**.
