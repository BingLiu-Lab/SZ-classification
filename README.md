# SZ Classification

## Introduction

This repository contains all the code for multisite Schizophrenia (SZ) Classification by integrating structural magnetic resonance imaging data (sMRI) with polygenic risk score (PGRS). Details of the method are described in the paper: [Multisite Schizophrenia Classification by Integrating Structural Magnetic Resonance Imaging Data with Polygenic Risk Score.]()

## Data

To protect participant privacy, which the genetic data used in our study may involve, the data is currently not available for public download. For specific research needs, the corresponding author can be contacted to discuss data sharing (bing.liu@bnu.edu.cn). 

## Usage

### 1. **Training SVM and LR models**

- SVM

  ```
  python code/SVM.py
  ```

  It will generate two folder. One is `model/` to store the trained models, and the other is `variable/`  to store the false positive rate (fpr) and true positive rate (tpr).

- LR

  ```
  python code/LR.py
  ```

### 2. **Ensemble learning**

```
python code/Ensemble_Learning.py
```

### 3. Features weight

```
python code/weight.py
```


















