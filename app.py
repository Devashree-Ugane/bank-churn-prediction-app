import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import shap
import sys  # Added to track active virtual environment packages Safely

# ================================
# SESSION STATE INIT
# ================================
if "predicted" not in st.session_state:
    st.session_state.predicted = False
if "input_data" not in st.session_state:
    st.session_state.input_data = None
if "prediction_prob" not in st.session_state:
    st.session_state.prediction_prob = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# ================================
# ABSOLUTE PATH HANDLING
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Ensure directories exist so logging doesn't fail
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ================================
# LOAD MODEL
# ================================
pipeline = None
pipeline_path = os.path.join(MODEL_DIR, "pipeline.pkl")
if os.path.exists(pipeline_path):
    with open(pipeline_path, "rb") as file:
        pipeline = pickle.load(file)

# ================================
# LOAD JSON FILES
# ================================
def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return None

data_stats = load_json(os.path.join(MODEL_DIR, "data_stats.json"))
metrics = load_json(os.path.join(MODEL_DIR, "metrics.json"))
model_comparison = load_json(os.path.join(MODEL_DIR, "model_comparison.json"))
retrain_history = load_json(os.path.join(MODEL_DIR, "retrain_history.json")) or []

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("🏦 Bank Customer Churn Prediction")

if pipeline is None:
    st.error("⚠️ `pipeline.pkl` model file not found or could not be loaded. Please go to the 'Retraining' tab and click 'Retrain Model Now' to generate it!")

# ================================
# TABS
# ================================
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Dashboard", "🔁 Retraining"])

# ================================
# TAB 1: PREDICT
# ================================
with tab1:
    st.sidebar.header("Customer Information")

    def user_input_features():
        country = st.sidebar.selectbox("Country", ["France", "Spain", "Germany"])
        gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
        credit_score = st.sidebar.number_input("Credit Score", 300, 850, 650)
        age = st.sidebar.number_input("Age", 18, 100, 30)
        tenure = st.sidebar.number_input("Tenure", 0, 10, 3)
        balance = st.sidebar.number_input("Balance", 0.0, 200000.0, 50000.0)
        products_number = st.sidebar.number_input("Products", 1, 4, 1)
        credit_card = st.sidebar.selectbox("Has Credit Card", ["Yes", "No"])
        active_member = st.sidebar.selectbox("Active Member", ["Yes", "No"])
        estimated_salary = st.sidebar.number_input("Salary", 0.0, 200000.0, 50000.0)

        data = {
            "credit_score": credit_score,
            "country": country,
            "gender": gender,
            "age": age,
            "tenure": tenure,
            "balance": balance,
            "products_number": products_number,
            "credit_card": 1 if credit_card == "Yes" else 0,
            "active_member": 1 if active_member == "Yes" else 0,
            "estimated_salary": estimated_salary
        }
        return pd.DataFrame(data, index=[0])

    raw_input_data = user_input_features()
    input_data = raw_input_data.copy()

    # Align input columns dynamically with training data expected by your pipeline
    if pipeline is not None and hasattr(pipeline, "feature_names_in_"):
        try:
            input_data = input_data[pipeline.feature_names_in_]
        except Exception as e:
            st.warning(f"Column alignment warning: {e}")

    def clean_feature_name(name):
        name = name.replace("num__", "").replace("cat__", "")
        return name.replace("_", " ").title()

    if pipeline is not None:
        if st.button("Predict"):
            prediction_prob = pipeline.predict_proba(input_data)[0][1]
            prediction = pipeline.predict(input_data)[0]

            st.session_state.predicted = True
            st.session_state.input_data = input_data
            st.session_state.prediction_prob = prediction_prob
            st.session_state.prediction = prediction

            log_row = input_data.copy()
            log_row["prediction"] = prediction
            log_row["probability"] = prediction_prob

            log_path = os.path.join(DATA_DIR, "user_inputs.csv")
            try:
                existing = pd.read_csv(log_path)
                updated = pd.concat([existing, log_row], ignore_index=True)
            except:
                updated = log_row
            updated.to_csv(log_path, index=False)

        if st.session_state.predicted:
            input_data = st.session_state.input_data
            prediction_prob = st.session_state.prediction_prob
            prediction = st.session_state.prediction

            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Prediction")
                if prediction == 1:
                    st.error("⚠️ Customer is likely to churn")
                else:
                    st.success("✅ Customer will stay")

            with col2:
                st.subheader("Churn Probability")
                st.write(f"### {prediction_prob:.2%}")

            # ================================
            # WHAT-IF SIMULATOR
            # ================================
            st.divider()
            st.subheader("🎛️ What-If Simulator")

            new_balance = st.slider("Balance", 0, 200000, int(input_data['balance'].iloc[0]))
            new_age = st.slider("Age", 18, 100, int(input_data['age'].iloc[0]))
            new_credit = st.slider("Credit Score", 300, 850, int(input_data['credit_score'].iloc[0]))
            new_products = st.slider("Number of Products", 1, 4, int(input_data['products_number'].iloc[0]))

            simulated_data = input_data.copy()
            simulated_data['balance'] = new_balance
            simulated_data['age'] = new_age
            simulated_data['credit_score'] = new_credit
            simulated_data['products_number'] = new_products

            # Maintain correct column sequence for simulator execution
            if hasattr(pipeline, "feature_names_in_"):
                simulated_data = simulated_data[pipeline.feature_names_in_]

            new_prob = pipeline.predict_proba(simulated_data)[0][1]
            st.write(f"### New Churn Probability: {new_prob:.2%}")

            if new_prob > prediction_prob:
                st.warning("⚠️ Risk increased after changes")
            else:
                st.success("✅ Risk decreased after changes")

            # ================================
            # SHAP EXPLAINABILITY
            # ================================
            st.divider()
            st.subheader("🔍 SHAP Explainability — Why This Prediction?")

            try:
                model = pipeline.named_steps["classifier"]
                preprocessor_step = pipeline.named_steps["preprocessor"]
                X_transformed = preprocessor_step.transform(input_data)
                feature_names = preprocessor_step.get_feature_names_out()

                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_transformed)

                if isinstance(shap_values, list):
                    sv = shap_values[1][0]
                else:
                    sv = shap_values[0]

                shap_df = pd.DataFrame({
                    "Feature": [clean_feature_name(f) for f in feature_names],
                    "SHAP Value": sv,
                    "Impact": ["Increases churn risk" if v > 0 else "Decreases churn risk" for v in sv]
                })
                shap_df["Abs"] = shap_df["SHAP Value"].abs()
                shap_df = shap_df.sort_values("Abs", ascending=False).head(10).drop("Abs", axis=1)

                st.dataframe(
                    shap_df.style.background_gradient(subset=["SHAP Value"], cmap="RdYlGn_r"),
                    width="stretch"
                )
                st.caption("Positive SHAP = pushes toward churn | Negative SHAP = pushes toward staying")

            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")

            # ================================
            # FEATURE IMPORTANCE (FALLBACK)
            # ================================
            st.divider()
            st.subheader("📊 Top Factors Affecting Churn (Model-level)")

            try:
                model = pipeline.named_steps["classifier"]
                preprocessor_step = pipeline.named_steps["preprocessor"]
                feature_names = preprocessor_step.get_feature_names_out()
                importances = model.feature_importances_

                feat_imp = pd.Series(importances, index=feature_names)
                feat_imp = feat_imp.sort_values(ascending=False).head(10)
                feat_imp.index = [clean_feature_name(i) for i in feat_imp.index]
                st.bar_chart(feat_imp)
            except:
                st.info("Feature importance not available for this model type.")

            # ================================
            # DRIFT DETECTION
            # ================================
            st.divider()
            st.subheader("🌊 Data Quality & Drift Check")

            if data_stats:
                drift_details = []
                for col in input_data.columns:
                    if col in data_stats and isinstance(data_stats[col], dict):
                        stats = data_stats[col]
                        val = input_data[col].iloc[0]
                        mean = stats.get("mean", 0)
                        std = stats.get("std", 0)
                        p5 = stats.get("p5", mean - 2*std)
                        p95 = stats.get("p95", mean + 2*std)

                        z_score = abs(val - mean) / std if std > 0 else 0
                        out_of_range = val < p5 or val > p95

                        status = "✅ Normal"
                        if z_score > 3:
                            status = "🔴 Extreme outlier (>3σ)"
                        elif z_score > 2:
                            status = "🟡 Moderate deviation (>2σ)"
                        elif out_of_range:
                            status = "🟠 Outside training range (P5–P95)"

                        drift_details.append({
                            "Feature": col.replace("_", " ").title(),
                            "Your Value": round(float(val), 2),
                            "Training Mean": round(mean, 2),
                            "Z-Score": round(z_score, 2),
                            "Status": status
                        })

                drift_df = pd.DataFrame(drift_details)
                st.dataframe(drift_df, width="stretch")

                red_flags = drift_df[drift_df["Status"].str.startswith("🔴")]
                if not red_flags.empty:
                    st.error("⚠️ Extreme drift detected in: " + ", ".join(red_flags["Feature"].tolist()))
                else:
                    st.success("✅ No extreme drift detected")

            # ================================
            # RECOMMENDATIONS
            # ================================
            st.divider()
            st.subheader("💡 Recommended Next Steps")

            if prediction_prob < 0.3:
                st.write("Customer is low risk. Maintain engagement.")
            else:
                if input_data['active_member'].iloc[0] == 0:
                    st.write("• Increase engagement (inactive customer)")
                if input_data['products_number'].iloc[0] == 1:
                    st.write("• Cross-sell more products")
                if input_data['balance'].iloc[0] < 1000:
                    st.write("• Encourage higher balance")
                if input_data['credit_score'].iloc[0] < 600:
                    st.write("• Offer financial advisory")

# ================================
# TAB 2: MODEL DASHBOARD
# ================================
with tab2:
    st.header("📊 Model Performance Dashboard")

    if metrics:
        st.subheader(f"Current Best Model: `{metrics.get('best_model', 'N/A')}`")
        st.caption(f"Last trained: {metrics.get('trained_at', 'Unknown')}")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
        col2.metric("Precision", f"{metrics.get('precision', 0):.2%}")
        col3.metric("Recall", f"{metrics.get('recall', 0):.2%}")
        col4.metric("F1 Score", f"{metrics.get('f1', 0):.2%}")
        col5.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.2%}")

        st.divider()
        st.subheader("Confusion Matrix")
        cm = metrics.get("confusion_matrix")
        if cm:
            cm_df = pd.DataFrame(
                cm,
                index=["Actual: Stay", "Actual: Churn"],
                columns=["Predicted: Stay", "Predicted: Churn"]
            )
            st.dataframe(cm_df.style.background_gradient(cmap="Blues"), width="content")

            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            st.caption(f"True Negatives: {tn} | False Positives: {fp} | False Negatives: {fn} | True Positives: {tp}")
    else:
        st.warning("No metrics found. Please run retraining first.")

    if model_comparison:
        st.divider()
        st.subheader("🏆 Model Comparison")
        comp_df = pd.DataFrame(model_comparison).T
        comp_df.index.name = "Model"
        comp_df = comp_df.reset_index()

        numeric_cols_comp = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        st.dataframe(
            comp_df.style.highlight_max(subset=numeric_cols_comp, color="#d4edda"),
            width="stretch"
        )
        st.caption("Green = best value in each column")

# ================================
# TAB 3: RETRAINING
# ================================
with tab3:
    st.header("🔁 Model Maintenance & Retraining")

    if st.button("🔄 Retrain Model Now"):
        with st.spinner("Retraining model... this may take a minute"):
            # Fixed runtime execution using active environment paths
            train_script = os.path.join(BASE_DIR, "train.py")
            result = os.system(f"{sys.executable} {train_script}")
            if result == 0:
                st.success("✅ Model retrained successfully! Please reboot or reload the page to apply new pipelines.")
                st.rerun()
            else:
                st.error("❌ Retraining failed. Check Streamlit logs for execution details.")

    st.divider()
    st.subheader("📈 Retraining History")

    if retrain_history:
        history_df = pd.DataFrame(retrain_history)
        history_df = history_df.sort_values("timestamp", ascending=False)
        st.dataframe(history_df, width="stretch")
        st.line_chart(history_df.set_index("timestamp")[["roc_auc", "f1", "accuracy"]])

    st.divider()
    st.subheader("📋 Recent Predictions Log")

    try:
        log_path = os.path.join(DATA_DIR, "user_inputs.csv")
        user_log = pd.read_csv(log_path)
        st.write(f"Total predictions made: **{len(user_log)}**")
        st.dataframe(user_log.tail(20), width="stretch")
    except:
        st.info("No prediction log yet.")
