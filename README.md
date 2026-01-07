# 🤖 Probabilistic AI Assessment System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?style=for-the-badge&logo=numpy)

**Graduate-Level Machine Learning Implementation**
**Probabilistic Classification with Bayesian Networks & Gaussian Processes**

![Accuracy](https://img.shields.io/badge/Accuracy-87--100%25-brightgreen?style=flat-square)
![Datasets](https://img.shields.io/badge/Datasets-50K%2B%20Records-blue?style=flat-square)
![Grade](https://img.shields.io/badge/Module%20Weight-50%25-orange?style=flat-square)

</div>

---

## 🎯 Project Overview

This project represents **distinction-level graduate work** in probabilistic artificial intelligence, implementing and comparing three advanced machine learning approaches for binary classification across real-world datasets.

### 📚 Academic Context

- **Module**: CMP9794 - Advanced Artificial Intelligence
- **Institution**: University of Lincoln, MSc Computer Science
- **Assessment Weight**: 50% of module grade
- **Level**: Graduate (Master's)
- **Achievement**: Distinction-level performance

### 🎓 Learning Objectives Demonstrated

✅ **Probabilistic Reasoning**: Implement probabilistic inference methods
✅ **Algorithm Comparison**: Critically evaluate different ML approaches
✅ **Real-World Application**: Apply theory to fraud detection and medical diagnosis
✅ **Computational Complexity**: Analyze and optimize algorithm efficiency
✅ **Experimental Design**: Proper train-test splitting and evaluation metrics

---

## 🧠 Algorithms Implemented

### 1. **Discrete Bayesian Networks** (Structure Learning)

**Approach**: Chow-Liu Algorithm for Maximum Spanning Tree

**Key Techniques**:
- **Discretization**: KBinsDiscretizer with 3 bins (quantile strategy)
- **Structure Learning**: Chow-Liu algorithm with O(n²) complexity
- **Parameter Learning**: Maximum Likelihood Estimation (MLE)
- **Inference**: Variable Elimination

**Pros**: Captures variable dependencies, interpretable structure
**Cons**: Information loss from discretization, moderate accuracy

**Results**:
- Fraud Detection: 87.72% accuracy
- Heart Disease: 85.85% accuracy
- Training Time: 10-36 seconds

---

### 2. **Gaussian Bayesian Networks** (Continuous Variables)

**Approach**: Linear Gaussian Conditional Probability Distributions

**Key Techniques**:
- **CPDs**: Linear Gaussian models for continuous features
- **Structure**: Naive Bayes independence assumptions
- **Benefits**: No discretization, preserves continuous information
- **Efficiency**: Fast training (< 1 second)

**Pros**: Handles continuous data naturally, very fast training
**Cons**: Assumes linear relationships, Naive Bayes independence

**Results**:
- Fraud Detection: 88.40% accuracy
- Heart Disease: 82.93% accuracy
- Training Time: < 1 second

---

### 3. **Gaussian Processes** (Non-Parametric Method)

**Approach**: GP Classification with RBF Kernel

**Key Techniques**:
- **Kernel**: Radial Basis Function (RBF) with hyperparameter optimization
- **Feature Selection**: SelectKBest using mutual information (k=10)
- **Inference**: Laplace approximation for binary classification
- **Uncertainty**: Provides prediction confidence intervals

**Pros**: Highest accuracy, uncertainty quantification, non-parametric
**Cons**: High computational cost (O(n³)), longer training time

**Results**:
- Fraud Detection: **98.52% accuracy** 🏆
- Heart Disease: **100% accuracy** 🏆
- Training Time: 133-272 seconds

---

## 📊 Datasets

### 1. Credit Card Fraud Detection
- **Size**: 50,000 transactions
- **Features**: 21 numerical features
- **Class Distribution**: Imbalanced (fraud vs. legitimate)
- **Source**: Financial transaction data
- **Challenge**: Detecting fraudulent patterns in high-volume data

### 2. Heart Disease Diagnosis
- **Size**: 1,025 patient records
- **Features**: 14 clinical features (age, blood pressure, cholesterol, etc.)
- **Class Distribution**: Binary (disease present/absent)
- **Source**: Medical diagnostic data
- **Challenge**: Accurate disease prediction from clinical indicators

---

## 📈 Performance Comparison

| Method | Fraud Accuracy | Heart Accuracy | Training Time | Complexity |
|--------|----------------|----------------|---------------|------------|
| **Discrete BN** | 87.72% | 85.85% | 10-36s | O(n²) |
| **Gaussian BN** | 88.40% | 82.93% | < 1s | O(n) |
| **Gaussian Process** | **98.52%** 🏆 | **100%** 🏆 | 133-272s | O(n³) |

### Key Insights

1. **Accuracy vs. Speed Trade-off**: GP achieves highest accuracy but requires significantly more training time
2. **Gaussian BN**: Best for real-time applications requiring fast inference
3. **Discrete BN**: Good balance with interpretable structure learning
4. **Dataset Sensitivity**: GP shows consistent excellence across both domains

---

## 🛠️ Technologies & Libraries

### Core ML Stack
- **Python 3.11+**: Programming language
- **pgmpy**: Probabilistic graphical models library
- **scikit-learn**: Machine learning algorithms and utilities
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **SciPy**: Scientific computing and optimization

### Visualization & Analysis
- **Matplotlib**: Plotting and visualizations
- **Seaborn**: Statistical data visualization
- **12 PNG Visualizations**: Confusion matrices, ROC curves, feature importance

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC (Area Under Curve)
- Brier Score (probabilistic calibration)
- Confusion Matrices
- Feature Importance Analysis

---

## 🏗️ Project Structure

```
📁 CMP9794-Probabilistic-AI-Assessment/
├── 📄 README.md                          # This file
├── 📂 datasets/
│   ├── fraud_detection.csv              # 50,000 transactions
│   └── heart_disease.csv                # 1,025 patient records
├── 📂 src/
│   ├── 01_data_exploration.py           # EDA and visualization
│   ├── 02_discrete_bn.py                # Discrete Bayesian Network
│   ├── 03_gaussian_bn.py                # Gaussian Bayesian Network
│   ├── 04_gaussian_process.py           # Gaussian Process Classifier
│   ├── 05_model_comparison.py           # Comparative analysis
│   ├── helpers.py                       # Utility functions
│   └── preprocessing.py                 # Data preprocessing
├── 📂 results/
│   ├── 📊 visualizations/               # 12 PNG plots
│   ├── 📊 metrics/                      # 11 CSV performance files
│   └── 📄 comparison_report.csv         # Consolidated results
├── 📂 docs/
│   └── technical_report.pdf             # IEEE-format analysis
└── requirements.txt                      # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- 8GB RAM minimum (for Gaussian Process on large dataset)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/toptech5419/CMP9794-Probabilistic-AI-Assessment.git
cd CMP9794-Probabilistic-AI-Assessment
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Dependencies
```txt
pgmpy>=0.1.23
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
```

---

## 📖 Usage

### Run All Experiments
```bash
# Run complete analysis pipeline (~8-9 minutes)
python run_all.py
```

### Run Individual Methods

**1. Data Exploration**
```bash
python src/01_data_exploration.py
```
- Generates EDA visualizations
- Outputs: Feature distributions, correlation matrices

**2. Discrete Bayesian Network**
```bash
python src/02_discrete_bn.py
```
- Trains DBN with Chow-Liu structure
- Outputs: Accuracy metrics, confusion matrix

**3. Gaussian Bayesian Network**
```bash
python src/03_gaussian_bn.py
```
- Trains GBN with continuous CPDs
- Outputs: Performance metrics, feature importance

**4. Gaussian Process**
```bash
python src/04_gaussian_process.py
```
- Trains GP classifier with RBF kernel
- Outputs: Accuracy, uncertainty quantification

**5. Comparative Analysis**
```bash
python src/05_model_comparison.py
```
- Compares all three methods
- Outputs: Side-by-side performance metrics

---

## 📊 Results & Visualizations

### Generated Outputs

**12 Visualization Files**:
1. Fraud dataset feature distributions
2. Heart disease feature correlations
3. Discrete BN structure graph (fraud)
4. Discrete BN confusion matrix
5. Gaussian BN feature importance
6. Gaussian BN ROC curve
7. GP prediction confidence intervals
8. GP feature selection analysis
9. Method comparison bar charts
10. Accuracy vs. training time scatter
11. Brier score calibration plots
12. Error analysis by feature

**11 Metrics CSV Files**:
- Individual method performance on both datasets
- Cross-validation results
- Feature importance rankings
- Prediction probabilities
- Consolidated comparison table

---

## 🧪 Experimental Design

### Train-Test Split
- **Strategy**: Stratified 80/20 split
- **Justification**: Single split due to GP computational complexity (O(n³))
- **Reproducibility**: Random seed set for consistent results

### Evaluation Metrics
1. **Accuracy**: Overall correct predictions
2. **Precision**: True positives / (TP + FP)
3. **Recall**: True positives / (TP + FN)
4. **F1-Score**: Harmonic mean of precision and recall
5. **AUC-ROC**: Classifier discrimination ability
6. **Brier Score**: Probabilistic calibration quality

### Hyperparameter Optimization
- **Discrete BN**: Discretization bins (tested: 3, 5, 10)
- **Gaussian Process**: RBF kernel length scale (GridSearchCV)
- **Feature Selection**: Mutual information threshold (k=10 optimal)

---

## 🎓 Key Learnings & Insights

### 1. **Algorithm Selection Matters**
- **GP**: Best for accuracy-critical applications (medical diagnosis)
- **Gaussian BN**: Best for real-time systems (fraud detection dashboards)
- **Discrete BN**: Best for interpretability (explainable AI)

### 2. **Computational Complexity Trade-offs**
- O(n³) GP training limits scalability to ~50K records
- O(n²) Chow-Liu allows larger datasets with acceptable accuracy
- O(n) Gaussian BN enables real-time predictions

### 3. **Domain Considerations**
- **Medical**: Perfect accuracy (100%) worth longer training time
- **Fraud**: Fast inference more important than marginal accuracy gains

### 4. **Uncertainty Quantification**
- GP provides confidence intervals valuable for risk-sensitive applications
- Bayesian methods naturally handle uncertainty through probabilistic inference

---

## 🔬 Methodological Justifications

### Why Chow-Liu for Discrete BN?
- Optimal tree structure guaranteed
- O(n²) complexity scales to moderate datasets
- Captures pairwise dependencies without overfitting

### Why Naive Bayes Structure for Gaussian BN?
- Simplifies learning to linear time
- Strong independence assumptions reduce variance
- Sufficient for linearly separable classes

### Why RBF Kernel for GP?
- Universal approximator for smooth functions
- Captures non-linear patterns in medical data
- Hyperparameter optimization via GridSearchCV

### Why Single Train-Test Split?
- GP training on 50K records already requires 4+ minutes
- Cross-validation would multiply runtime 5-10x
- Stratified split ensures representative evaluation

---

## 📄 Academic Outputs

### Technical Report (IEEE Format)
- **Pages**: 15+ pages
- **Sections**:
  1. Introduction & Literature Review
  2. Methodology (3 algorithms described)
  3. Experimental Setup
  4. Results & Analysis
  5. Discussion & Conclusions
  6. References (15+ papers)

### Deliverables
✅ Complete implementation of 3 probabilistic methods
✅ Comprehensive evaluation on 2 datasets
✅ Visualizations and performance metrics
✅ Technical report with justifications
✅ Reproducible code with documentation

---

## 🏆 Achievement Highlights

- ✨ **100% Accuracy** on heart disease diagnosis (GP method)
- ✨ **98.52% Accuracy** on fraud detection (GP method)
- ✨ **Distinction-Level Performance** (top grade band)
- ✨ **Rigorous Methodology** with proper experimental design
- ✨ **Production-Quality Code** with clean architecture
- ✨ **Complete Documentation** including justifications

---

## 📚 References & Resources

### Key Papers Implemented
1. Chow & Liu (1968) - "Approximating discrete probability distributions with dependence trees"
2. Rasmussen & Williams (2006) - "Gaussian Processes for Machine Learning"
3. Friedman et al. (1997) - "Bayesian Network Classifiers"

### Libraries Documentation
- [pgmpy Documentation](https://pgmpy.org/)
- [scikit-learn GP Module](https://scikit-learn.org/stable/modules/gaussian_process.html)
- [Bayesian Networks Tutorial](https://www.bnlearn.com/)

---

## 🤝 Contributing

This is an academic project completed as coursework. While direct contributions are not accepted, you are welcome to:

- Fork the repository for your own learning
- Reference the methodologies in your work (with citation)
- Suggest improvements via issues

### Citation

If you use this work in your research, please cite:

```bibtex
@misc{alabi2025probabilistic,
  author = {Alabi, Temitope},
  title = {Probabilistic AI Assessment: Comparative Analysis of Bayesian Networks and Gaussian Processes},
  year = {2025},
  institution = {University of Lincoln},
  course = {CMP9794 - Advanced Artificial Intelligence}
}
```

---

## 📜 License

This project is submitted as academic coursework for CMP9794 at the University of Lincoln.

**Academic Integrity**: Code and methodologies are original work completed for assessment purposes.

---

## 👨‍💻 Author

**Temitope Alabi**
MSc Computer Science Student
University of Lincoln, UK

- 🌐 GitHub: [@toptech5419](https://github.com/toptech5419)
- 💼 LinkedIn: [toptech5419](https://linkedin.com/in/toptech5419)
- 📧 Email: alabitemitope51@gmail.com

### Academic Profile
- **Program**: MSc Computer Science (2025-2026)
- **Specialization**: Artificial Intelligence & Machine Learning
- **Previous Modules**:
  - CMP9794: Advanced Artificial Intelligence (This Project)
  - CMP9133: Programming Principles (C++ Distributed Systems)
  - Information Systems Security

---

## 🙏 Acknowledgments

- **Module Leader**: University of Lincoln School of Computer Science
- **Datasets**: UCI Machine Learning Repository
- **Libraries**: pgmpy, scikit-learn development teams
- **References**: Research papers cited in technical report

---

## 📸 Sample Visualizations

<table>
<tr>
<td width="50%">

### Confusion Matrix (GP - Fraud)
![Confusion Matrix](https://via.placeholder.com/400x400?text=Confusion+Matrix)

</td>
<td width="50%">

### ROC Curve Comparison
![ROC Curves](https://via.placeholder.com/400x400?text=ROC+Curves)

</td>
</tr>
<tr>
<td width="50%">

### Feature Importance (Gaussian BN)
![Feature Importance](https://via.placeholder.com/400x400?text=Feature+Importance)

</td>
<td width="50%">

### Algorithm Comparison
![Method Comparison](https://via.placeholder.com/400x400?text=Method+Comparison)

</td>
</tr>
</table>

---

<div align="center">

**Graduate-Level Machine Learning Implementation**
**University of Lincoln | MSc Computer Science**

[⬆ Back to Top](#-probabilistic-ai-assessment-system)

</div>
