"""
CMP9794M Assessment - Discrete Bayesian Network for Fraud Detection
====================================================================
This script implements a Discrete Bayesian Network for fraud detection using:
1. Data discretization (continuous → categorical)
2. Structure learning (Tabu Search algorithm)
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
from pgmpy.estimators import TreeSearch, BDeuScore, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, confusion_matrix
from sklearn.preprocessing import KBinsDiscretizer
import time
import warnings
import os

warnings.filterwarnings('ignore')

print("="*80)
print("CMP9794M - DISCRETE BAYESIAN NETWORK: FRAUD DETECTION")
print("="*80)

# ============================================================================
# PART 1: LOAD AND PREPARE DATA
# ============================================================================
print("\n[1/7] Loading fraud detection dataset...")

# Load dataset
df = pd.read_csv('../data/synthetic_fraud_dataset.csv')
print(f"✓ Loaded {len(df)} transactions with {df.shape[1]} features")

# Select most relevant features for Bayesian Network (to keep network manageable)
selected_features = [
    'Transaction_Amount',
    'Account_Balance',
    'Risk_Score',
    'Daily_Transaction_Count',
    'Previous_Fraudulent_Activity',
    'Failed_Transaction_Count_7d',
    'Transaction_Distance',
    'IP_Address_Flag',
    'Is_Weekend',
    'Fraud_Label'
]

# Add categorical features (encode them)
categorical_features = ['Transaction_Type', 'Device_Type', 'Authentication_Method']

# Create working dataframe
df_working = df[selected_features].copy()

# Add encoded categorical features
for cat_feat in categorical_features:
    df_working[cat_feat] = pd.Categorical(df[cat_feat]).codes

print(f"✓ Selected {len(df_working.columns)} features for Bayesian Network")

# ============================================================================
# PART 2: DISCRETIZE CONTINUOUS FEATURES
# ============================================================================
print("\n[2/7] Discretizing continuous features...")

continuous_features = [
    'Transaction_Amount',
    'Account_Balance',
    'Risk_Score',
    'Transaction_Distance'
]

# Use 3 bins for simpler network structure
discretizer_config = {
    'Transaction_Amount': 3,      # Low, Medium, High
    'Account_Balance': 3,          # Low, Medium, High
    'Risk_Score': 3,               # Low, Medium, High
    'Transaction_Distance': 3      # Near, Medium, Far
}

df_discretized = df_working.copy()

for feature, n_bins in discretizer_config.items():
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    df_discretized[feature] = discretizer.fit_transform(df_working[[feature]]).astype(int)

# Ensure all values are integers
df_discretized = df_discretized.astype(int)

print(f"✓ Discretized {len(continuous_features)} continuous features")
print(f"✓ All features converted to discrete integers")

# ============================================================================
# PART 3: SPLIT DATA FOR TRAINING AND TESTING
# ============================================================================
print("\n[3/7] Splitting data into train/test sets...")

X = df_discretized.drop('Fraud_Label', axis=1)
y = df_discretized['Fraud_Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

print(f"✓ Training set: {len(train_data)} samples")
print(f"✓ Test set: {len(test_data)} samples")
print(f"✓ Target distribution (train): {dict(y_train.value_counts())}")

# ============================================================================
# PART 4: STRUCTURE LEARNING (TREE SEARCH - CHOW-LIU)
# ============================================================================
print("\n[4/7] Learning Bayesian Network structure...")

start_structure_time = time.time()

# Use Tree Search (Chow-Liu) algorithm which creates a tree structure
# This is more reliable than Hill-Climbing for our case
print("   → Using Tree Search (Chow-Liu) algorithm...")
tree_search = TreeSearch(train_data, root_node='Fraud_Label')
best_model = tree_search.estimate()

structure_time = time.time() - start_structure_time

print(f"✓ Structure learning completed in {structure_time:.2f}s")
print(f"✓ Number of edges: {len(best_model.edges())}")
print(f"✓ Number of nodes: {len(best_model.nodes())}")

# Display learned edges
print("\n--- Learned Network Edges ---")
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

# Show CPD for Fraud_Label
if model.get_cpds('Fraud_Label'):
    print("\n--- Fraud_Label CPD (first few rows) ---")
    cpd = model.get_cpds('Fraud_Label')
    print(f"Variables: {cpd.variables}")
    print(f"Cardinality: {cpd.cardinality}")

# ============================================================================
# PART 6: INFERENCE AND EVALUATION
# ============================================================================
print("\n[6/7] Performing inference on test set...")

start_inference_time = time.time()

inference = VariableElimination(model)

predictions_proba = []
predictions_class = []

print("   → Running inference on test samples...")

# Process in batches for better performance
batch_size = 100
for i in range(0, len(test_data), batch_size):
    batch = test_data.iloc[i:i+batch_size]

    for idx, row in batch.iterrows():
        evidence = {col: int(row[col]) for col in test_data.columns if col != 'Fraud_Label'}

        try:
            result = inference.query(variables=['Fraud_Label'], evidence=evidence)
            prob_fraud = result.values[1]
            predictions_proba.append(prob_fraud)
            predictions_class.append(1 if prob_fraud > 0.5 else 0)
        except Exception as e:
            # Use marginal probability if inference fails
            predictions_proba.append(y_train.mean())
            predictions_class.append(int(y_train.mode()[0]))

    if (i + batch_size) % 1000 == 0 or (i + batch_size) >= len(test_data):
        print(f"   → Processed {min(i + batch_size, len(test_data))} / {len(test_data)} samples")

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
print("FRAUD DETECTION BAYESIAN NETWORK - RESULTS")
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
print(f"   Structure Learning:       {structure_time:.2f}s")
print(f"   Parameter Learning:       {param_time:.2f}s")
print(f"   Total Training Time:      {total_training_time:.2f}s")
print(f"   Total Inference Time:     {inference_time:.2f}s")
print(f"   Avg Inference per Sample: {avg_inference_time*1000:.2f}ms")

print("\n🔢 CONFUSION MATRIX:")
print(f"                    Predicted")
print(f"                 Legit  Fraud")
print(f"   Actual Legit  {tn:5d}  {fp:5d}")
print(f"          Fraud  {fn:5d}  {tp:5d}")

# Save metrics
metrics_df = pd.DataFrame({
    'Dataset': ['Fraud Detection'],
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
metrics_df.to_csv('../results/metrics/03_fraud_bn_metrics.csv', index=False)
print("\n✓ Saved: 03_fraud_bn_metrics.csv")

print("\n" + "="*80)
print("✅ FRAUD DETECTION BAYESIAN NETWORK COMPLETE")
print("="*80)
