"""
CMP9794M Assessment - Gaussian Process for Fraud Detection
===========================================================
This script implements a Gaussian Process for fraud detection using:
1. Feature selection (GP is computationally expensive)
2. RBF (Radial Basis Function) kernel
3. Uncertainty quantification
4. Probabilistic prediction

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
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import time
import warnings
import os

warnings.filterwarnings('ignore')

print("="*80)
print("CMP9794M - GAUSSIAN PROCESS: FRAUD DETECTION")
print("="*80)

# ============================================================================
# PART 1: LOAD AND PREPARE DATA
# ============================================================================
print("\n[1/7] Loading fraud detection dataset...")

# Load dataset
df = pd.read_csv('../data/synthetic_fraud_dataset.csv')
print(f"✓ Loaded {len(df)} transactions with {df.shape[1]} features")

# Select continuous features (GP works best with continuous data)
continuous_features = [
    'Transaction_Amount',
    'Account_Balance',
    'Risk_Score',
    'Transaction_Distance',
    'Daily_Transaction_Count',
    'Failed_Transaction_Count_7d',
    'Previous_Fraudulent_Activity',
    'IP_Address_Flag',
    'Is_Weekend'
]

# Create working dataframe
df_working = df[continuous_features + ['Fraud_Label']].copy()

print(f"✓ Selected {len(continuous_features)} continuous features")

# ============================================================================
# PART 2: FEATURE SELECTION (GP IS SLOW WITH MANY FEATURES)
# ============================================================================
print("\n[2/7] Performing feature selection...")

X = df_working[continuous_features]
y = df_working['Fraud_Label']

# Use mutual information to select top K most informative features
# GP is O(n^3) complexity, so we need to limit features
k_features = 7  # Select top 7 features for computational efficiency

selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
X_selected = selector.fit_transform(X, y)

# Get selected feature names
selected_mask = selector.get_support()
selected_features = [feat for feat, selected in zip(continuous_features, selected_mask) if selected]

print(f"✓ Selected {k_features} most informative features using Mutual Information")
print("\n--- Selected Features (by importance) ---")
feature_scores = selector.scores_
for feat, score, selected in sorted(zip(continuous_features, feature_scores, selected_mask),
                                   key=lambda x: x[1], reverse=True):
    marker = "✓" if selected else " "
    print(f"   [{marker}] {feat:35s} Score: {score:.4f}")

# ============================================================================
# PART 3: STANDARDIZE AND SPLIT DATA
# ============================================================================
print("\n[3/7] Standardizing and splitting data...")

# Standardize features (important for GP)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# GP has O(n^3) complexity - use subset for training to avoid memory issues
# Use stratified sampling to maintain class balance
max_train_samples = 1500  # Limit to 1500 samples for computational tractability

if len(X_train) > max_train_samples:
    from sklearn.model_selection import StratifiedShuffleSplit

    # Convert y_train to numpy array for proper indexing
    y_train_np = np.array(y_train)

    sss = StratifiedShuffleSplit(n_splits=1, train_size=max_train_samples, random_state=42)
    for train_idx, _ in sss.split(X_train, y_train_np):
        X_train = X_train[train_idx]
        y_train = y_train_np[train_idx]

    print(f"✓ Reduced training set to {len(X_train)} samples (for computational efficiency)")

print(f"✓ Training set: {len(X_train)} samples × {k_features} features")
print(f"✓ Test set: {len(X_test)} samples × {k_features} features")
print(f"✓ Target distribution (train): {dict(pd.Series(y_train).value_counts())}")
print(f"   Note: GP requires O(n³) memory/time, limited to {max_train_samples} samples")

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
# Note: max_iter_predict limits the number of iterations for prediction convergence
gpc = GaussianProcessClassifier(
    kernel=kernel,
    random_state=42,
    n_restarts_optimizer=10,  # Number of restarts for optimizer
    max_iter_predict=100       # Max iterations for prediction
)

print("   → Training with Laplace approximation...")
print("   → This may take a few minutes on large datasets...")

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
predictions_proba = gpc.predict_proba(X_test)[:, 1]  # Probability of fraud (class 1)

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
print("GAUSSIAN PROCESS - FRAUD DETECTION RESULTS")
print("="*80)

print("\n📊 PERFORMANCE METRICS:")
print(f"   Accuracy:        {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   AUC-ROC:         {auc_roc:.4f}")
print(f"   Brier Score:     {brier_score:.4f} (lower is better)")
print(f"   Precision:       {precision:.4f}")
print(f"   Recall:          {recall:.4f}")
print(f"   F1-Score:        {f1_score:.4f}")
print(f"   Specificity:     {specificity:.4f}")

print("\n⏱️  COMPUTATIONAL EFFICIENCY:")
print(f"   Model Definition:         {structure_time:.2f}s")
print(f"   Training Time:            {train_time:.2f}s")
print(f"   Total Training Time:      {total_training_time:.2f}s")
print(f"   Total Inference Time:     {inference_time:.2f}s")
print(f"   Avg Inference per Sample: {avg_inference_time*1000:.2f}ms")

print("\n🔢 CONFUSION MATRIX:")
print(f"                    Predicted")
print(f"                 Legit  Fraud")
print(f"   Actual Legit  {tn:5d}  {fp:5d}")
print(f"          Fraud  {fn:5d}  {tp:5d}")

print("\n🎯 UNCERTAINTY QUANTIFICATION (UNIQUE TO GP):")
print(f"   High confidence predictions (p<0.3 or p>0.7): {high_confidence} ({high_confidence/len(y_test)*100:.1f}%)")
print(f"   Uncertain predictions (0.3<p<0.7):            {uncertain_samples} ({uncertain_samples/len(y_test)*100:.1f}%)")
print(f"   Mean prediction probability:                   {predictions_proba.mean():.4f}")
print(f"   Std prediction probability:                    {predictions_proba.std():.4f}")

# Comparison with other methods
print("\n📈 COMPARISON WITH OTHER METHODS:")
print(f"   Discrete BN Accuracy:  87.72%")
print(f"   Gaussian BN Accuracy:  88.40%")
print(f"   Gaussian Process Acc:  {accuracy*100:.2f}%")

# Save metrics
metrics_df = pd.DataFrame({
    'Dataset': ['Fraud Detection'],
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
    'Num_Features': [k_features],
    'Uncertain_Predictions': [uncertain_samples],
    'High_Confidence_Predictions': [high_confidence]
})

os.makedirs('../results/metrics', exist_ok=True)
metrics_df.to_csv('../results/metrics/07_gaussian_process_fraud_metrics.csv', index=False)
print("\n✓ Saved: 07_gaussian_process_fraud_metrics.csv")

print("\n" + "="*80)
print("✅ GAUSSIAN PROCESS - FRAUD DETECTION COMPLETE")
print("="*80)
print("\n🔬 KEY INSIGHTS:")
print("   1. GP provides uncertainty quantification (unique advantage)")
print("   2. Non-parametric approach captures complex non-linear relationships")
print("   3. Feature selection essential for computational tractability")
print("   4. Higher computational cost but provides prediction confidence")
print("="*80)
