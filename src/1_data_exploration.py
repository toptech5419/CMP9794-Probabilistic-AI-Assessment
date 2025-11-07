"""
CMP9794M Assessment - Data Exploration
======================================
This script performs exploratory data analysis on both datasets:
1. Fraud Detection (50,000 transactions)
2. Heart Disease (1,025 patients)

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
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# Set style for professional-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Create output directories if they don't exist
os.makedirs('../results/figures', exist_ok=True)
os.makedirs('../results/metrics', exist_ok=True)

print("="*80)
print("CMP9794M - DATA EXPLORATION")
print("="*80)

# ============================================================================
# PART 1: LOAD DATASETS
# ============================================================================
print("\n[1/5] Loading datasets...")

# Load fraud detection dataset
fraud_df = pd.read_csv('../data/synthetic_fraud_dataset.csv')
print(f"✓ Fraud dataset loaded: {fraud_df.shape[0]} rows, {fraud_df.shape[1]} columns")

# Load heart disease dataset
heart_df = pd.read_csv('../data/heart.csv')
print(f"✓ Heart dataset loaded: {heart_df.shape[0]} rows, {heart_df.shape[1]} columns")

# ============================================================================
# PART 2: FRAUD DETECTION DATASET EXPLORATION
# ============================================================================
print("\n" + "="*80)
print("FRAUD DETECTION DATASET ANALYSIS")
print("="*80)

print("\n[2/5] Analyzing fraud detection data...")

# Basic info
print("\n--- Dataset Info ---")
print(f"Shape: {fraud_df.shape}")
print(f"Memory usage: {fraud_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Display first few rows
print("\n--- First 5 rows ---")
print(fraud_df.head())

# Data types
print("\n--- Feature Types ---")
print(fraud_df.dtypes)

# Missing values
print("\n--- Missing Values ---")
missing = fraud_df.isnull().sum()
if missing.sum() == 0:
    print("✓ No missing values found")
else:
    print(missing[missing > 0])

# Class distribution
print("\n--- Target Variable Distribution (Fraud_Label) ---")
fraud_counts = fraud_df['Fraud_Label'].value_counts()
print(fraud_counts)
print(f"\nClass balance: {fraud_counts[0]} legitimate ({fraud_counts[0]/len(fraud_df)*100:.1f}%), "
      f"{fraud_counts[1]} fraud ({fraud_counts[1]/len(fraud_df)*100:.1f}%)")

# Identify feature types
print("\n--- Feature Classification ---")
fraud_numeric = fraud_df.select_dtypes(include=[np.number]).columns.tolist()
fraud_categorical = fraud_df.select_dtypes(include=['object']).columns.tolist()

print(f"Numeric features ({len(fraud_numeric)}): {fraud_numeric[:5]}...")
print(f"Categorical features ({len(fraud_categorical)}): {fraud_categorical}")

# Descriptive statistics for key numeric features
print("\n--- Descriptive Statistics (Key Numeric Features) ---")
key_fraud_features = ['Transaction_Amount', 'Account_Balance', 'Risk_Score',
                      'Daily_Transaction_Count', 'Transaction_Distance']
print(fraud_df[key_fraud_features].describe())

# Save fraud dataset summary
fraud_summary = {
    'Dataset': 'Fraud Detection',
    'Total_Instances': len(fraud_df),
    'Features': fraud_df.shape[1] - 1,  # excluding target
    'Numeric_Features': len(fraud_numeric) - 1,  # excluding target
    'Categorical_Features': len(fraud_categorical),
    'Class_0_Count': fraud_counts[0],
    'Class_1_Count': fraud_counts[1],
    'Class_Imbalance_Ratio': fraud_counts[0] / fraud_counts[1],
    'Missing_Values': missing.sum()
}

# ============================================================================
# PART 3: HEART DISEASE DATASET EXPLORATION
# ============================================================================
print("\n" + "="*80)
print("HEART DISEASE DATASET ANALYSIS")
print("="*80)

print("\n[3/5] Analyzing heart disease data...")

# Basic info
print("\n--- Dataset Info ---")
print(f"Shape: {heart_df.shape}")
print(f"Memory usage: {heart_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Display first few rows
print("\n--- First 5 rows ---")
print(heart_df.head())

# Data types
print("\n--- Feature Types ---")
print(heart_df.dtypes)

# Missing values
print("\n--- Missing Values ---")
missing_heart = heart_df.isnull().sum()
if missing_heart.sum() == 0:
    print("✓ No missing values found")
else:
    print(missing_heart[missing_heart > 0])

# Class distribution
print("\n--- Target Variable Distribution (target) ---")
heart_counts = heart_df['target'].value_counts()
print(heart_counts)
print(f"\nClass balance: {heart_counts[0]} no disease ({heart_counts[0]/len(heart_df)*100:.1f}%), "
      f"{heart_counts[1]} disease ({heart_counts[1]/len(heart_df)*100:.1f}%)")

# Feature names mapping (for clarity)
feature_names = {
    'age': 'Age',
    'sex': 'Sex (1=male, 0=female)',
    'cp': 'Chest Pain Type',
    'trestbps': 'Resting Blood Pressure',
    'chol': 'Cholesterol',
    'fbs': 'Fasting Blood Sugar > 120',
    'restecg': 'Resting ECG',
    'thalach': 'Max Heart Rate',
    'exang': 'Exercise Induced Angina',
    'oldpeak': 'ST Depression',
    'slope': 'Slope of ST Segment',
    'ca': 'Number of Vessels (Fluoroscopy)',
    'thal': 'Thallium Stress Test',
    'target': 'Heart Disease (0=no, 1=yes)'
}

print("\n--- Feature Descriptions ---")
for col, desc in feature_names.items():
    if col in heart_df.columns:
        print(f"{col:12s} : {desc}")

# Identify feature types
print("\n--- Feature Classification ---")
heart_continuous = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
heart_discrete = [col for col in heart_df.columns if col not in heart_continuous and col != 'target']

print(f"Continuous features ({len(heart_continuous)}): {heart_continuous}")
print(f"Discrete features ({len(heart_discrete)}): {heart_discrete}")

# Descriptive statistics
print("\n--- Descriptive Statistics (Continuous Features) ---")
print(heart_df[heart_continuous].describe())

# Save heart dataset summary
heart_summary = {
    'Dataset': 'Heart Disease',
    'Total_Instances': len(heart_df),
    'Features': heart_df.shape[1] - 1,
    'Continuous_Features': len(heart_continuous),
    'Discrete_Features': len(heart_discrete),
    'Class_0_Count': heart_counts[0],
    'Class_1_Count': heart_counts[1],
    'Class_Imbalance_Ratio': heart_counts[0] / heart_counts[1] if heart_counts[1] > 0 else 0,
    'Missing_Values': missing_heart.sum()
}

# ============================================================================
# PART 4: VISUALIZATIONS
# ============================================================================
print("\n[4/5] Creating visualizations...")

# --- FRAUD DATASET VISUALIZATIONS ---

# 1. Class distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Fraud class distribution
fraud_counts.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
axes[0].set_title('Fraud Detection - Class Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Class (0=Legitimate, 1=Fraud)')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(['Legitimate', 'Fraud'], rotation=0)
for i, v in enumerate(fraud_counts):
    axes[0].text(i, v + 500, str(v), ha='center', fontweight='bold')

# Heart class distribution
heart_counts.plot(kind='bar', ax=axes[1], color=['#3498db', '#e67e22'])
axes[1].set_title('Heart Disease - Class Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Class (0=No Disease, 1=Disease)')
axes[1].set_ylabel('Count')
axes[1].set_xticklabels(['No Disease', 'Disease'], rotation=0)
for i, v in enumerate(heart_counts):
    axes[1].text(i, v + 10, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('../results/figures/01_class_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_class_distributions.png")
plt.close()

# 2. Fraud - Key numeric features distribution
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()

key_fraud_viz = ['Transaction_Amount', 'Account_Balance', 'Risk_Score',
                 'Daily_Transaction_Count', 'Transaction_Distance', 'Card_Age']

for idx, feature in enumerate(key_fraud_viz):
    if feature in fraud_df.columns:
        fraud_df[feature].hist(bins=30, ax=axes[idx], color='#3498db', edgecolor='black')
        axes[idx].set_title(f'{feature}', fontweight='bold')
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Frequency')

plt.suptitle('Fraud Dataset - Key Numeric Features Distribution', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../results/figures/02_fraud_feature_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_fraud_feature_distributions.png")
plt.close()

# 3. Heart - Continuous features distribution by target
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()

for idx, feature in enumerate(heart_continuous):
    heart_df[heart_df['target'] == 0][feature].hist(bins=20, ax=axes[idx], alpha=0.6,
                                                      label='No Disease', color='#3498db')
    heart_df[heart_df['target'] == 1][feature].hist(bins=20, ax=axes[idx], alpha=0.6,
                                                      label='Disease', color='#e67e22')
    axes[idx].set_title(f'{feature}', fontweight='bold')
    axes[idx].set_xlabel('Value')
    axes[idx].set_ylabel('Frequency')
    axes[idx].legend()

# Remove extra subplot
if len(heart_continuous) < 6:
    fig.delaxes(axes[5])

plt.suptitle('Heart Dataset - Continuous Features by Disease Status', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../results/figures/03_heart_feature_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_heart_feature_distributions.png")
plt.close()

# 4. Correlation heatmaps
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Fraud correlation (select numeric columns only)
fraud_numeric_cols = fraud_df.select_dtypes(include=[np.number]).columns
fraud_corr = fraud_df[fraud_numeric_cols].corr()
sns.heatmap(fraud_corr, cmap='coolwarm', center=0, ax=axes[0],
            cbar_kws={'label': 'Correlation'}, square=False)
axes[0].set_title('Fraud Dataset - Feature Correlations', fontsize=14, fontweight='bold')

# Heart correlation
heart_corr = heart_df.corr()
sns.heatmap(heart_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            ax=axes[1], cbar_kws={'label': 'Correlation'}, square=True)
axes[1].set_title('Heart Dataset - Feature Correlations', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('../results/figures/04_correlation_heatmaps.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_correlation_heatmaps.png")
plt.close()

# ============================================================================
# PART 5: SAVE SUMMARY STATISTICS
# ============================================================================
print("\n[5/5] Saving summary statistics...")

# Combine summaries
summary_df = pd.DataFrame([fraud_summary, heart_summary])
summary_df.to_csv('../results/metrics/00_dataset_summary.csv', index=False)
print("✓ Saved: 00_dataset_summary.csv")

# Save detailed statistics
fraud_stats = fraud_df[key_fraud_features].describe()
fraud_stats.to_csv('../results/metrics/01_fraud_statistics.csv')
print("✓ Saved: 01_fraud_statistics.csv")

heart_stats = heart_df[heart_continuous].describe()
heart_stats.to_csv('../results/metrics/02_heart_statistics.csv')
print("✓ Saved: 02_heart_statistics.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("DATA EXPLORATION COMPLETE")
print("="*80)

print("\n📊 SUMMARY:")
print(f"\n1. Fraud Detection Dataset:")
print(f"   - {fraud_df.shape[0]:,} transactions, {fraud_df.shape[1]} features")
print(f"   - Class imbalance: {fraud_summary['Class_Imbalance_Ratio']:.2f}:1 (legitimate:fraud)")
print(f"   - {len(fraud_numeric)} numeric features, {len(fraud_categorical)} categorical")

print(f"\n2. Heart Disease Dataset:")
print(f"   - {heart_df.shape[0]:,} patients, {heart_df.shape[1]} features")
print(f"   - Class imbalance: {heart_summary['Class_Imbalance_Ratio']:.2f}:1 (no disease:disease)")
print(f"   - {len(heart_continuous)} continuous features, {len(heart_discrete)} discrete")

print(f"\n📁 Output Files:")
print(f"   - 4 visualization plots → results/figures/")
print(f"   - 3 summary CSV files → results/metrics/")

print("="*80)
