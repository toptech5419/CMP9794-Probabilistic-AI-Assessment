"""
CMP9794M Assessment - Discrete Bayesian Network for Heart Disease
==================================================================
This script implements a Discrete Bayesian Network for heart disease diagnosis using:
1. Data discretization (continuous → categorical)
2. Structure learning (Tree Search algorithm)
3. Parameter learning (Maximum Likelihood Estimation)
4. Probabilistic inference

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
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import TreeSearch, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, confusion_matrix
from sklearn.preprocessing import KBinsDiscretizer
import time
import warnings
import os

warnings.filterwarnings('ignore')

print("="*80)
print("CMP9794M - DISCRETE BAYESIAN NETWORK: HEART DISEASE DIAGNOSIS")
print("="*80)

# ============================================================================
# PART 1: LOAD AND PREPARE DATA
# ============================================================================
print("\n[1/7] Loading heart disease dataset...")

# Load dataset
df = pd.read_csv('../data/heart.csv')
print(f"✓ Loaded {len(df)} patients with {df.shape[1]} features")

# All features are already numeric, but we need to discretize continuous ones
# Continuous features: age, trestbps, chol, thalach, oldpeak
# Discrete features: sex, cp, fbs, restecg, exang, slope, ca, thal
# Target: target (0=no disease, 1=disease)

print(f"✓ Dataset has {df.shape[1]-1} features + 1 target variable")

# ============================================================================
# PART 2: DISCRETIZE CONTINUOUS FEATURES
# ============================================================================
print("\n[2/7] Discretizing continuous features...")

# Identify continuous features that need discretization
continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Discretization configuration
discretizer_config = {
    'age': 3,          # Young, Middle-aged, Elderly
    'trestbps': 3,     # Low, Normal, High blood pressure
    'chol': 3,         # Low, Normal, High cholesterol
    'thalach': 3,      # Low, Normal, High max heart rate
    'oldpeak': 3       # Low, Medium, High ST depression
}

df_discretized = df.copy()

for feature, n_bins in discretizer_config.items():
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    df_discretized[feature] = discretizer.fit_transform(df[[feature]]).astype(int)

# Ensure all values are integers
df_discretized = df_discretized.astype(int)

print(f"✓ Discretized {len(continuous_features)} continuous features")
print(f"✓ All {df_discretized.shape[1]} features are now discrete")

# Display feature statistics
print("\n--- Feature Cardinalities ---")
for col in df_discretized.columns:
    n_unique = df_discretized[col].nunique()
    print(f"   {col:20s}: {n_unique} unique values")

# ============================================================================
# PART 3: SPLIT DATA FOR TRAINING AND TESTING
# ============================================================================
print("\n[3/7] Splitting data into train/test sets...")

X = df_discretized.drop('target', axis=1)
y = df_discretized['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

print(f"✓ Training set: {len(train_data)} samples")
print(f"✓ Test set: {len(test_data)} samples")
print(f"✓ Target distribution (train): {dict(y_train.value_counts())}")
print(f"   No disease: {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train)*100:.1f}%)")
print(f"   Disease:    {(y_train==1).sum()} ({(y_train==1).sum()/len(y_train)*100:.1f}%)")

# ============================================================================
# PART 4: STRUCTURE LEARNING (TREE SEARCH - CHOW-LIU)
# ============================================================================
print("\n[4/7] Learning Bayesian Network structure...")

start_structure_time = time.time()

# Use Tree Search (Chow-Liu) algorithm with target as root
print("   → Using Tree Search (Chow-Liu) algorithm...")
print("   → Building tree with 'target' as root node...")
tree_search = TreeSearch(train_data, root_node='target')
best_model = tree_search.estimate()

structure_time = time.time() - start_structure_time

print(f"✓ Structure learning completed in {structure_time:.2f}s")
print(f"✓ Number of edges: {len(best_model.edges())}")
print(f"✓ Number of nodes: {len(best_model.nodes())}")

# Display learned edges
print("\n--- Learned Network Structure ---")
edges_list = list(best_model.edges())
for parent, child in edges_list:
    print(f"   {parent} → {child}")

# ============================================================================
# PART 5: PARAMETER LEARNING
# ============================================================================
print("\n[5/7] Learning Bayesian Network parameters...")

start_param_time = time.time()

model = BayesianNetwork(best_model.edges())
model.fit(train_data, estimator=MaximumLikelihoodEstimator)

param_time = time.time() - start_param_time
total_training_time = structure_time + param_time

print(f"✓ Parameter learning completed in {param_time:.2f}s")
print(f"✓ Total training time: {total_training_time:.2f}s")

# Show CPD for target variable
if model.get_cpds('target'):
    print("\n--- Target Variable CPD ---")
    cpd = model.get_cpds('target')
    print(f"Variables: {cpd.variables}")
    print(f"Cardinality: {cpd.cardinality}")
    print("Probability distribution:")
    print(cpd)

# ============================================================================
# PART 6: INFERENCE AND EVALUATION
# ============================================================================
print("\n[6/7] Performing inference on test set...")

start_inference_time = time.time()

inference = VariableElimination(model)

predictions_proba = []
predictions_class = []

print("   → Running inference on test samples...")

for idx, row in test_data.iterrows():
    evidence = {col: int(row[col]) for col in test_data.columns if col != 'target'}

    try:
        result = inference.query(variables=['target'], evidence=evidence)
        prob_disease = result.values[1]  # Probability of disease (class 1)
        predictions_proba.append(prob_disease)
        predictions_class.append(1 if prob_disease > 0.5 else 0)
    except Exception as e:
        # Use marginal probability if inference fails
        predictions_proba.append(y_train.mean())
        predictions_class.append(int(y_train.mode()[0]))

inference_time = time.time() - start_inference_time
avg_inference_time = inference_time / len(test_data)

print(f"✓ Inference completed in {inference_time:.2f}s")
print(f"✓ Average inference time per sample: {avg_inference_time*1000:.2f}ms")

# ============================================================================
# PART 7: CALCULATE METRICS
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

print("\n" + "="*80)
print("HEART DISEASE BAYESIAN NETWORK - RESULTS")
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
print(f"   Structure Learning:       {structure_time:.2f}s")
print(f"   Parameter Learning:       {param_time:.2f}s")
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

# Save metrics
metrics_df = pd.DataFrame({
    'Dataset': ['Heart Disease'],
    'Model': ['Discrete Bayesian Network (Tree)'],
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
    'Num_Edges': [len(best_model.edges())],
    'Num_Nodes': [len(best_model.nodes())]
})

os.makedirs('../results/metrics', exist_ok=True)
metrics_df.to_csv('../results/metrics/04_heart_bn_metrics.csv', index=False)
print("\n✓ Saved: 04_heart_bn_metrics.csv")

print("\n" + "="*80)
print("✅ HEART DISEASE BAYESIAN NETWORK COMPLETE")
print("="*80)
print("="*80)
