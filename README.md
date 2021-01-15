# SZ Classification

## Introduction

This repository contains code and data for multisite Schizophrenia (SZ) Classification by integrating structural magnetic resonance imaging data (sMRI) and polygenic risk score (PGRS). Details of the method are described in the article: [Multisite Schizophrenia Classification by integrating structural magnetic resonance imaging data and polygenic risk score.]()

## Data

246 dimensional average gray matter volume (GMV) features based sMRI data and 145 dimensional PGRS features were used in this study. The features are stored in `data/all_data.csv`. The first column is the subjects ID, where ‘NC’ are normal controls and ‘SZ’ are SZ patients. Columns 2-247 are GMV features and columns 248-392 are PGRS features. Site information for each subject is stored in `data/Sites.csv`. 

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

















