# FaAlGrad: Fairness through Alignment of Gradients across Different Subpopulations

This repository contains the implementation of the Fairness-Aware Algorithm for Gradient Alignment (FaAlGrad), accepted at Transactions on Machine Learning Research (TMLR). This work explores fairness in machine learning through meta-learning techniques.

## Overview of the Paper

This work aims to handle the bias in machine learning models and enhance their fairness by aligning the loss gradients. Specifically, leveraging the meta-learning technique, we propose a novel training framework that aligns the gradients computed across different subpopulations for learning fair classifiers. Our experiments on multiple benchmark datasets demonstrate significant improvements in fairness metrics without having any exclusive regularizers for fairness. 

Below is a visual representation of the proposed framework:
![FaAlGrad](FaAlGrad_Diagram.png)

## Requirements

- Python 3.9 or lower (not compatible with Python 3.11+ due to library restrictions).
- Ensure `pip` and `conda` are installed on your system.

Install necessary libraries by running:
```bash
pip install -r requirements.txt
```

## Setup

1) Clone the repository: 
```python
git clone https://github.com/NikitaMalik2303/FaAlGrad.git
cd FaAlGrad
```
2) Set up a virtual environment:
```python
conda create -n faalgrad python=3.9
conda activate faalgrad
```
3) Install dependencies:
```python
pip install -r requirements.txt
```

## Usage

Run the training script using the following command:
```python
python main.py --split_mode <config> --split_ratio <value>
```

## Arguments:

1) **--split_mode**: Select one of the following configurations:

- **config_1** - Assign protected group to support set, unprotected to query set.

- **config_2** - Xs : protected 0's + unprotected 1's and Xq : protected 1's + unprotected 0's

- **varying_proportions** - the ratio of protected samples in Xs is varied, and Xq consists of remaining protected samples and all unprotected samples 

- **split_case_swapping** - the ratio of unprotected samples in Xs is varied, and Xq encompassed the remaining unprotected samples and all protected samples.

2) **--split_ratio**: (Optional) split of the training data between the two subsets: Xq and Xs. (0.5 =< split_ratio =< 0.8)




