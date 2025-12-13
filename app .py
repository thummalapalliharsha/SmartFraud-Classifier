import streamlit as st
import numpy as np
import pandas as pd
import joblib
import gdown
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="SmartFraud Classifer System", layout="centered")

st.title("SmartFraud Classifer System")
st.write("This system predicts whether a credit card transaction is *Legitimate* or *Fraudulent* using Tree-based Machine Learning Models.")
st.markdown("---")

FILE_ID = "1BPcVzWYi7uL1VkmNjP_wqHJlWVAsypE6"
DATA_PATH = "creditcard.csv"

if not os.path.exists(DATA_PATH):
    gdown.download(
        f"https://drive.google.com/uc?id={FILE_ID}",
        DATA_PATH,
        quiet=False
    )

try:
    data = pd.read_csv(DATA_PATH)
    features = [col for col in data.columns if col.lower() != "class"]
    expected_len = len(features)
    st.success(f"Loaded dataset successfully with {expected_len} features.")
except Exception as e:
    st.error(f"Unable to load dataset: {e}")
    st.stop()
    
models = {}
metrics = {}

def load_model(name, filename, train_acc=None, test_acc=None, auc=None, precision=None, recall=None):
    if os.path.exists(filename):
        try:
            model = joblib.load(filename)
            models[name] = model
            metrics[name] = {
                "train_acc": train_acc,
                "test_acc": test_acc,
                "auc": auc,
                "precision": precision,
                "recall": recall
            }
        except Exception as e:
            st.warning(f"Failed to load {filename}: {e}")

load_model("Decision Tree", "decision_tree_model.pkl", train_acc=0.97, test_acc=0.93, auc=96.3, precision=0.94, recall=0.99)
load_model("Random Forest", "random_forest_model.pkl", train_acc=0.99, test_acc=0.96, auc=98.3, precision=0.97, recall=0.98)
load_model("XGBoost", "xgboost_model.pkl", train_acc=0.99, test_acc=0.98, auc=98.7, precision=0.98, recall=0.99)

if not models:
    st.error("No model files found! Please place .pkl files in the same folder.")
    st.stop()

st.subheader("Model Performance")

comp_df = pd.DataFrame(metrics).T
comp_df_display = comp_df.copy()

for col in ["train_acc", "test_acc", "precision", "recall"]:
    if col in comp_df_display.columns:
        comp_df_display[col] = comp_df_display[col].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "-")

if "auc" in comp_df_display.columns:
    comp_df_display["auc"] = comp_df_display["auc"].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "-")

st.table(comp_df_display)
st.markdown("---")

model_choice = st.selectbox("Select Model for Prediction", list(models.keys()))
model = models[model_choice]
m = metrics.get(model_choice, {})

st.markdown(f"Model Selected: *{model_choice}*")

col1, col2 = st.columns(2)
with col1:
    st.metric("Training Accuracy", f"{m['train_acc']*100:.2f}%")
    st.metric("Testing Accuracy", f"{m['test_acc']*100:.2f}%")
with col2:
    st.metric("AUC Score", f"{m['auc']:.3f}")
    st.metric("Precision / Recall", f"{m['precision']:.2f} / {m['recall']:.2f}")

st.markdown("---")

st.subheader("Enter Transaction Details")
st.write("Enter all feature values separated by commas in the order below:")
st.code(", ".join(features), language="text")

example_input = "0.0, -1.3598, 1.1918, -0.358, 1.549, -0.225, -0.854, 0.896, 0.185, -1.188, 0.121, 0.023, -0.224, 0.222, 0.015, -0.082, 0.034, -0.002, 0.004, 0.014, -0.004, 0.001, 0.02, -0.018, 0.006, 0.0, 0.01, 0.01, 378.66, 0"
st.text_area("Example Input", example_input, height=90)

user_input = st.text_area("Enter your transaction data (comma-separated):", "", height=100)
st.markdown("---")

if st.button("Predict Transaction Status"):
    try:
        values = np.array([list(map(float, user_input.split(",")))])
        if values.shape[1] != expected_len:
            st.error(f"⚠ Incorrect number of features. Expected {expected_len}, got {values.shape[1]}.")
        else:
            prediction = model.predict(values)[0]
            probs = model.predict_proba(values)[0]
            prob_legit = probs[0]
            prob_fraud = probs[1]

            if prediction == 1:
                st.error(" Fraudulent Transaction Detected!")
            else:
                st.success(" Legitimate Transaction")

            st.markdown("Prediction Probabilities")
            st.write(f"*Legitimate Probability:* {prob_legit*100:.2f}%")
            st.write(f"*Fraud Probability:* {prob_fraud*100:.2f}%")

            fig, ax = plt.subplots(figsize=(6, 1))
            classes = ['Legitimate', 'Fraud']
            probs = [prob_legit, prob_fraud]
            colors = ['#27AE60', '#E74C3C']

            ax.barh(classes, probs, color=colors)
            for i, v in enumerate(probs):
                ax.text(v + 0.01, i, f"{v*100:.2f}%", color='black', va='center')
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability')
            ax.set_title('Fraud Detection Probability')
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Invalid input: {e}")

st.markdown("---")
st.caption("Developed by E2Group | SmartFraud Classifier | Streamlit Deployment © 2025")

