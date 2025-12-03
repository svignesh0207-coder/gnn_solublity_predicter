import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.data import Data
from rdkit import Chem
import numpy as np

# ------------------- Model Definition -------------------
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


# ------------------- SMILES → Graph Conversion -------------------
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features (ESOL format)
    x = []
    for atom in mol.GetAtoms():
        x.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetTotalNumHs(),
            int(atom.IsInRing()),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
            atom.GetMass() / 100,
            1.0
        ])
    x = torch.tensor(x, dtype=torch.float)

    # Edges
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices += [[i, j], [j, i]]  # undirected graph
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    # Batch vector (all nodes belong to graph 0)
    batch = torch.zeros(x.shape[0], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, batch=batch)


# ------------------- Streamlit App -------------------
st.set_page_config(page_title="GNN Solubility Predictor", layout="centered")
st.title("🧪 Molecular Solubility Predictor (logS)")
st.markdown("Graph Neural Network (GCN) trained on **ESOL** dataset to predict aqueous solubility.")


smiles = st.text_input(
    "Enter SMILES string:",
    value="CCO",
    help="Examples: Ethanol = CCO, Benzene = c1ccccc1"
)

if st.button("Predict Solubility", type="primary"):
    with st.spinner("Processing molecule & running GNN..."):
        graph = smiles_to_graph(smiles)
        if graph is None:
            st.error("❌ Invalid SMILES string!")
        else:
            with torch.no_grad():
                pred = model(graph.x, graph.edge_index, graph.batch)
            logS = pred.item()

            st.success(f"**Predicted logS = {logS:.3f}**")

            # Interpretations
            if logS > 0:
                st.info("💧 Highly soluble")
            elif logS > -2:
                st.info("🧪 Moderately soluble")
            elif logS > -4:
                st.warning("⚠️ Poorly soluble")
            else:
                st.error("❗ Very poorly soluble / insoluble")

            st.markdown(
                f"**Approx. solubility:** {10**logS:.2e} mol/L"
            )

st.markdown("---")
st.caption("Built by Vignesh • GNN using PyTorch Geometric • Trained on ESOL dataset")
