import streamlit as st
import matplotlib.pyplot as plt
import torch

# import model components
from model import (
    model, preprocess, le,
    SVM_ACC, LOG_ACC, NN_ACC,
    X, answer_map
)

st.title("Customer Service Ticket Classification App")
st.write("ANN-based prediction with recommended response message.")

# --------------------- INPUT FIELDS ---------------------
ticket_text = st.text_area("Enter Ticket Text:")

priority = st.selectbox("Priority:", X['priority'].unique())
queue = st.selectbox("Queue:", X['queue'].unique())
main_tag = st.selectbox("Main Tag:", X['main_tag'].unique())
sub_tag = st.selectbox("Sub Tag:", X['sub_tag'].unique())
extra_tag = st.selectbox("Extra Tag:", X['extra_tag'].unique())

# --------------------- PREDICTION ---------------------
if st.button("Predict Category (ANN)"):

    X_new = preprocess(ticket_text, priority, queue, main_tag, sub_tag, extra_tag)

    # ANN prediction
    X_t = torch.tensor(X_new, dtype=torch.float32)
    pred_ann = torch.argmax(model(X_t), dim=1).item()

    category = le.inverse_transform([pred_ann])[0]

    st.subheader("✔ Final Prediction")
    st.success(f"Predicted Category: **{category}**")

    # ---------------- ANSWER OUTPUT ----------------
    st.subheader("💬 Recommended Response")
    st.info(answer_map.get(category, "No predefined answer available."))

    # ---------------- DETAILS ----------------
    st.subheader("📌 Ticket Details")
    st.write(f"**Ticket Text:** {ticket_text}")
    st.write(f"**Priority:** {priority}")
    st.write(f"**Queue:** {queue}")
    st.write(f"**Main Tag:** {main_tag}")
    st.write(f"**Sub Tag:** {sub_tag}")
    st.write(f"**Extra Tag:** {extra_tag}")

# ---------------------- ACCURACY GRAPH ----------------------
st.header("📊 Model Accuracy Comparison")

fig, ax = plt.subplots(figsize=(7,5))

models = ["SVM", "Logistic Regression", "ANN"]
accuracies = [SVM_ACC, LOG_ACC, NN_ACC]

bars = ax.bar(models, accuracies)

for bar in bars:
    bar.set_edgecolor("black")
    bar.set_linewidth(1.5)

ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Model Performance Comparison", fontsize=14)
ax.grid(axis='y', linestyle="--", alpha=0.5)

for i, v in enumerate(accuracies):
    ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=12, fontweight='bold')

st.pyplot(fig)
