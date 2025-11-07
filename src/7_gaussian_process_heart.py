"""
CMP9794M Assessment - Gaussian Process for Heart Disease
=========================================================
This script implements a Gaussian Process for heart disease diagnosis using:
1. RBF (Radial Basis Function) kernel
2. Uncertainty quantification
3. Probabilistic prediction
4. All features (heart dataset is small enough)

Author: Alabi Temitope (Student ID: 30292576)
Date: October 2025
"""

import sys
import io
# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time
import warnings
import os

warnings.filterwarnings('ignore')

print("="*80)
print("CMP9794M - GAUSSIAN PROCESS: HEART DISEASE DIAGNOSIS")
print("="*80)

# ============================================================================
# PART 1: LOAD AND PREPARE DATA
# ============================================================================
print("\n[1/7] Loading heart disease dataset...")

# Load dataset
df = pd.read_csv('../data/heart.csv')
print(f"✓ Loaded {len(df)} patients with {df.shape[1]} features")

# Use all features (dataset is small enough for GP)
all_features = [col for col in df.columns if col != 'target']

# Create working dataframe
df_working = df.copy()

print(f"✓ Using ALL {len(all_features)} features")

# ============================================================================
# PART 2: STANDARDIZE FEATURES
# ============================================================================
print("\n[2/7] Standardizing features...")

# Standardize features (important for GP with RBF kernel)
scaler = StandardScaler()
X = scaler.fit_transform(df_working[all_features])
y = df_working['target'].values

print(f"✓ Standardized {len(all_features)} features")

# ============================================================================
# PART 3: SPLIT DATA
# ============================================================================
print("\n[3/7] Splitting data into train/test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Training set: {len(X_train)} samples × {len(all_features)} features")
print(f"✓ Test set: {len(X_test)} samples × {len(all_features)} features")
print(f"✓ Target distribution (train): {dict(pd.Series(y_train).value_counts())}")
print(f"   No disease: {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train)*100:.1f}%)")
print(f"   Disease:    {(y_train==1).sum()} ({(y_train==1).sum()/len(y_train)*100:.1f}%)")

# ============================================================================
# PART 4: DEFINE GAUSSIAN PROCESS MODEL
# ============================================================================
print("\n[4/7] Defining Gaussian Process model...")

start_structure_time = time.time()

# Define RBF kernel with optimizable hyperparameters
# kernel = C(constant_value, bounds) * RBF(length_scale, bounds)
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

print("   → Using RBF (Radial Basis Function) kernel")
print("   → Kernel: C(1.0) * RBF(length_scale=1.0)")
print("   → Hyperparameters will be optimized during training")

structure_time = time.time() - start_structure_time

print(f"✓ Model definition completed in {structure_time:.2f}s")

# ============================================================================
# PART 5: TRAIN GAUSSIAN PROCESS
# ============================================================================
print("\n[5/7] Training Gaussian Process...")

start_train_time = time.time()

# Initialize Gaussian Process Classifier
gpc = GaussianProcessClassifier(
    kernel=kernel,
    random_state=42,
    n_restarts_optimizer=10,  # Number of restarts for optimizer
    max_iter_predict=100       # Max iterations for prediction
)

print("   → Training with Laplace approximation...")

# Fit the model
gpc.fit(X_train, y_train)

train_time = time.time() - start_train_time
total_training_time = structure_time + train_time

print(f"✓ Training completed in {train_time:.2f}s")
print(f"✓ Total training time: {total_training_time:.2f}s")

# Show learned kernel parameters
print("\n--- Learned Kernel Parameters ---")
print(f"   Optimized kernel: {gpc.kernel_}")
print(f"   Log-marginal-likelihood: {gpc.log_marginal_likelihood():.4f}")

# ============================================================================
# PART 6: INFERENCE WITH UNCERTAINTY QUANTIFICATION
# ============================================================================
print("\n[6/7] Performing inference with uncertainty quantification...")

start_inference_time = time.time()

# Predict probabilities (this gives us uncertainty estimates)
print("   → Predicting class probabilities...")
predictions_proba = gpc.predict_proba(X_test)[:, 1]  # Probability of disease (class 1)

# Predict classes
predictions_class = gpc.predict(X_test)

inference_time = time.time() - start_inference_time
avg_inference_time = inference_time / len(X_test)

print(f"✓ Inference completed in {inference_time:.2f}s")
print(f"✓ Average inference time per sample: {avg_inference_time*1000:.2f}ms")

# ============================================================================
# PART 7: CALCULATE METRICS AND UNCERTAINTY ANALYSIS
# ============================================================================
print("\n[7/7] Calculating performance metrics...")

accuracy = accuracy_score(y_test, predictions_class)
auc_roc = roc_auc_score(y_test, predictions_proba)
brier_score = brier_score_loss(y_test, predictions_proba)

cm = confusion_matrix(y_test, predictions_class)
tn, fp, fn, tp = cm.ravel()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

# Uncertainty analysis
uncertain_samples = np.sum((predictions_proba > 0.3) & (predictions_proba < 0.7))
high_confidence = np.sum((predictions_proba < 0.3) | (predictions_proba > 0.7))

print("\n" + "="*80)
print("GAUSSIAN PROCESS - HEART DISEASE RESULTS")
print("="*80)

print("\n📊 PERFORMANCE METRICS:")
print(f"   Accuracy:        {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   AUC-ROC:         {auc_roc:.4f}")
print(f"   Brier Score:     {brier_score:.4f} (lower is better)")
print(f"   Precision:       {precision:.4f}")
print(f"   Recall:          {recall:.4f} (Sensitivity)")
print(f"   F1-Score:        {f1_score:.4f}")
print(f"   Specificity:     {specificity:.4f}")

print("\n⏱️  COMPUTATIONAL EFFICIENCY:")
print(f"   Model Definition:         {structure_time:.2f}s")
print(f"   Training Time:            {train_time:.2f}s")
print(f"   Total Training Time:      {total_training_time:.2f}s")
print(f"   Total Inference Time:     {inference_time:.2f}s")
print(f"   Avg Inference per Sample: {avg_inference_time*1000:.2f}ms")

print("\n🔢 CONFUSION MATRIX:")
print(f"                      Predicted")
print(f"                 No Disease  Disease")
print(f"   Actual No Dis   {tn:5d}      {fp:5d}")
print(f"          Disease  {fn:5d}      {tp:5d}")

# Clinical interpretation
print("\n🏥 CLINICAL INTERPRETATION:")
print(f"   True Negatives:  {tn} (correctly identified healthy)")
print(f"   False Positives: {fp} (healthy misclassified as diseased)")
print(f"   False Negatives: {fn} (diseased misclassified as healthy) ⚠️ Critical!")
print(f"   True Positives:  {tp} (correctly identified diseased)")

print("\n🎯 UNCERTAINTY QUANTIFICATION (UNIQUE TO GP):")
print(f"   High confidence predictions (p<0.3 or p>0.7): {high_confidence} ({high_confidence/len(y_test)*100:.1f}%)")
print(f"   Uncertain predictions (0.3<p<0.7):            {uncertain_samples} ({uncertain_samples/len(y_test)*100:.1f}%)")
print(f"   Mean prediction probability:                   {predictions_proba.mean():.4f}")
print(f"   Std prediction probability:                    {predictions_proba.std():.4f}")

# Comparison with other methods
print("\n📈 COMPARISON WITH OTHER METHODS:")
print(f"   Discrete BN Accuracy:  85.85%")
print(f"   Gaussian BN Accuracy:  82.93%")
print(f"   Gaussian Process Acc:  {accuracy*100:.2f}%")

# Save metrics
metrics_df = pd.DataFrame({
    'Dataset': ['Heart Disease'],
    'Model': ['Gaussian Process (RBF Kernel)'],
    'Accuracy': [accuracy],
    'AUC_ROC': [auc_roc],
    'Brier_Score': [brier_score],
    'Precision': [precision],
    'Recall': [recall],
    'F1_Score': [f1_score],
    'Specificity': [specificity],
    'Training_Time_s': [total_training_time],
    'Inference_Time_s': [inference_time],
    'Avg_Inference_ms': [avg_inference_time * 1000],
    'Num_Features': [len(all_features)],
    'Uncertain_Predictions': [uncertain_samples],
    'High_Confidence_Predictions': [high_confidence]
})

os.makedirs('../results/metrics', exist_ok=True)
metrics_df.to_csv('../results/metrics/08_gaussian_process_heart_metrics.csv', index=False)
print("\n✓ Saved: 08_gaussian_process_heart_metrics.csv")

print("\n" + "="*80)
print("✅ GAUSSIAN PROCESS - HEART DISEASE COMPLETE")
print("="*80)
print("\n🔬 KEY INSIGHTS:")
print("   1. GP provides uncertainty quantification for clinical decision support")
print("   2. Non-parametric approach captures complex patient patterns")
print("   3. Smaller dataset allows use of all 13 features")
print("   4. Uncertainty estimates help identify borderline cases needing further testing")
print("="*80)
