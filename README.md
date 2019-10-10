This is the MATLAB implementation of the paper [Link Prediction Under Imperfect Detection: Collaborative Filtering for Ecological Networks](https://arxiv.org/pdf/1910.03659.pdf).



# How to run

## With Synthetic Data
```
main_sim
```

## With Pollination Data
### Case 1: Semi-Real Data Evaluation with known parameters (detection probabilities)
- with constant detection probability
```
main_semi_const("ppi") or main_semi_const("hpi")
```
- with detection probability rom species traits
```
main_semi_traits("ppi") or main_semi_traits("hpi")
```

### Case 2: Real Data Evaluation with 10-fold cross-validation 
```
main_folds("ppi", 2)
```
1: RMSE, 2: rRMSE, 3: AUROC, 4:AUPRC

### Case 3: Real Data Evaluation with actual missing entries
```
main_missing
```
