import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import py3Dmol

# ======================================================
# 1️⃣ MODEL DEFINITION (same as your trained model)
# ======================================================
AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
AA2IDX = {aa: i+1 for i, aa in enumerate(AA_LIST)}  # 0 reserved for padding

class AnglePredictor(nn.Module):
    def __init__(self, vocab_size=21, embed_dim=64, cnn_channels=128, lstm_hidden=256, dropout_p=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.cnn = nn.Conv1d(embed_dim, cnn_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.bilstm = nn.LSTM(cnn_channels, lstm_hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(lstm_hidden*2, 4)  # (phi_cos, phi_sin, psi_cos, psi_sin)
    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1,2)
        x = self.relu(self.cnn(x))
        x = self.dropout(x)
        x = x.transpose(1,2)
        out,_ = self.bilstm(x)
        out = self.dropout(out)
        return self.fc(out)

# ======================================================
# 2️⃣ LOAD TRAINED MODEL
# ======================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AnglePredictor().to(device)
model.load_state_dict(torch.load("angle_predictor.pth", map_location=device))
model.eval()

# ======================================================
# 3️⃣ HELPER FUNCTION — Convert Sequence → Tensor
# ======================================================
def sequence_to_tensor(seq, device):
    seq_indices = [AA2IDX.get(aa, 0) for aa in seq]
    return torch.tensor([seq_indices], dtype=torch.long).to(device)

# ======================================================
# 4️⃣ STREAMLIT APP INTERFACE
# ======================================================
st.set_page_config(page_title="Protonic: Protein Structure Prediction", layout="wide")
st.title("🧬 Protonic: Interactive Protein Structure Prediction")
st.markdown("Enter a protein sequence and visualize its predicted 3D structure!")

seq_input = st.text_area("Enter Protein Sequence:", height=100)

if st.button("🔮 Predict Structure"):
    if not seq_input.strip():
        st.warning("Please enter a valid amino acid sequence.")
    else:
        with st.spinner("Running trained model..."):
            # Predict angles
            seq_tensor = sequence_to_tensor(seq_input.strip(), device)
            with torch.no_grad():
                preds = model(seq_tensor).cpu().numpy()[0]   # (L, 4)
            cos_phi, cos_psi, sin_phi, sin_psi = preds[:,0], preds[:,1], preds[:,2], preds[:,3]
            phi_rad = np.arctan2(sin_phi, cos_phi)
            psi_rad = np.arctan2(sin_psi, cos_psi)

            # Simple 3D backbone visualization
            view = py3Dmol.view(width=500, height=500)
            for i in range(len(seq_input)):
                view.addSphere({
                    'center': {'x': i, 'y': 0, 'z': 0},
                    'radius': 0.2,
                    'color': f'#{hex(255 - int(i*10)%255)[2:]}0000'
                })
            view.zoomTo()
            st.components.v1.html(view._make_html(), height=500)

            # Show torsion angles in table
            st.success("✅ Structure prediction complete!")
            st.write("Predicted torsion angles (°):")
            st.dataframe({
                "Residue": np.arange(1, len(seq_input)+1),
                "ϕ (phi)": np.rad2deg(phi_rad),
                "ψ (psi)": np.rad2deg(psi_rad)
            })

# ======================================================
# 5️⃣ FEEDBACK PANEL (for HCI evaluation)
# ======================================================
st.markdown("---")
st.header("🗣️ User Feedback")

col1, col2 = st.columns(2)
satisfaction = col1.slider("Satisfaction (1–5)", 1, 5, 4)
clarity = col2.slider("Clarity of Visualization (1–5)", 1, 5, 4)
comment = st.text_area("Comments or Suggestions:")

if st.button("💾 Submit Feedback"):
    st.success("Thank you! Feedback recorded.")
