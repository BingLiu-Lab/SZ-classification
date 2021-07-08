# SZ Classification

## Introduction

This repository contains all the code for multisite Schizophrenia (SZ) Classification by integrating structural magnetic resonance imaging data (sMRI) with polygenic risk score (PGRS). Details of the method are described in the article: [Multisite Schizophrenia Classification by Integrating Structural Magnetic Resonance Imaging Data with Polygenic Risk Score.]()

## Data

To protect participant privacy, which the genetic data used in our study may involve, the data is currently not available for public download. For specific research needs, the corresponding author can be contacted to discuss data sharing (bing.liu@bnu.edu.cn). 

## Usage

### 1. Data for leave-one-site-out cross-validation(CV)

In order to facilitate the operation of the code in this project, we divided the data in advance according to leave-one-site-out CV. The data were stored in `data/data_for_CV/`

### 2. **Training SVM and LR models**

- SVM

  ```
  python code/SVM.py
  ```

  It will generate two folder. One is `model/` to store the trained models, and the other is `variable/`  to store the false positive rate (fpr) and true positive rate (tpr).

- LR

  ```
  python code/LR.py
  ```

### 3. **Ensemble learning**

```
python code/Ensemble_Learning.py
```

### 4. Features weight

```
python code/weight.py
```

### 5. ROC curves

```
python code/ROC_curve.py
```

















