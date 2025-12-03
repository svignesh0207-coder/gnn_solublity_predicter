import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.data import Data
from rdkit import Chem
import numpy as np

# ------------------- Model Definition (exact same as training) -------------------
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        embedding_size = 64
        self.initial_conv = GCNConv(9, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)
        self.dropout = torch.nn.Dropout(0.3)
        self.out = torch.nn.Linear(embedding_size * 2, 1)

    def forward(self, x, edge_index, batch):
        h = F.tanh(self.initial_conv(x, edge_index))
        h = F.tanh(self.conv1(h, edge_index))
        h = F.tanh(self.conv2(h, edge_index))
        h = F.tanh(self.conv3(h, edge_index))
        h = self.dropout(h)
        h = torch.cat([gmp(h, batch), gap(h, batch)], dim=1)
        return self.out(h)

# ------------------- Load Model -------------------
@st.cache_resource
def load_model():
    model = GCN()
    model.load_state_dict(torch.load("best_gnn_esol.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ------------------- SMILES → Graph -------------------
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features (same as MoleculeNet ESOL)
    x = []
    for atom in mol.GetAtoms():
        x.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetTotalNumHs(),
            atom.IsInRing(),
            atom.GetHybridization(),
            atom.GetIsAromatic(),
            atom.GetMass() / 100,  # scaled
            1.0
        ])
    x = torch.tensor(x, dtype=torch.float)

    # Edges
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices += [[i, j], [j, i]]
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="GNN Solubility Predictor", layout="centered")
st.title("Molecular Solubility Predictor (logS)")
st.markdown("**Graph Neural Network trained on ESOL dataset** • RMSE ≈ 0.70–0.90")

smiles = st.text_input("Enter SMILES string", value="CCO", help="Example: Ethanol = CCO, Benzene = c1ccccc1")

if st.button("Predict Solubility", type="primary"):
    with st.spinner("Converting molecule & running GNN..."):
        graph = smiles_to_graph(smiles)
        if graph is None:
            st.error("Invalid SMILES string!")
        else:
            with torch.no_grad():
                pred = model(graph.x, graph.edge_index, torch.zeros(1, dtype=torch.long))
            logS = pred.item()
            st.success(f"**Predicted logS = {logS:.3f}**")
            if logS > 0:
                st.info("Highly soluble")
            elif logS > -2:
                st.info("Moderately soluble")
            elif logS > -4:
                st.warning("Poorly soluble")
            else:
                st.error("Very poorly soluble / insoluble")
            
            st.markdown(f"**Interpretation**: logS = {logS:.3f} → aqueous solubility ≈ {10**logS:.2e} mol/L")

st.markdown("---")
st.caption("Built by Vignesh • GNN from scratch using PyTorch Geometric • Model trained on ESOL dataset")