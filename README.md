# Molecular Solubility Predictor (logS) – Graph Neural Network

Predicts aqueous solubility (logS) of any small molecule from its SMILES string using a Graph Convolutional Network built from scratch with PyTorch Geometric.

### Key points
- Trained on the standard **ESOL (Delaney) dataset** (1 128 compounds)  
- 4-layer GCN with global mean + max pooling  
- No pre-trained models, no hand-crafted descriptors — pure message passing on molecular graphs  
- Fully interactive web app (Streamlit)

### Performance 
| Metric       | Value  | Note                                 |
|--------------|--------|--------------------------------------|
| Test RMSE    | 0.893  | Very solid for a simple from-scratch GCN |
| Test R²      | 0.805  | Comparable to/better than many XGBoost + ECFP baselines |

### Try it now
- Ethanol → `CCO`  
- Benzene → `c1ccccc1`  
- Caffeine → `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`

### Tech stack
- PyTorch + PyTorch Geometric  
- RDKit (featurization)  
- Streamlit (deployment)
