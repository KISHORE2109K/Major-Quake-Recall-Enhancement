# ============================================================
#  EARTHQUAKE RECALL IMPROVEMENT – RESEARCH-ONLY VERSION
#  (NO PREDICTION UI, RUNS ONLY ONCE)
# ============================================================

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import folium
from streamlit_folium import st_folium

# --------------------------------------------------------------------
# SESSION STATE – Run pipeline only once
# --------------------------------------------------------------------
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "training_columns" not in st.session_state:
    st.session_state.training_columns = None

st.set_page_config(page_title="Earthquake Recall Improvement", 
                   page_icon="🌋",
                   layout="wide")

# --------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------
st.markdown("""
<div style="background-color:#0E76A8;padding:20px;border-radius:10px">
<h1 style="color:white;text-align:center;">🌍 Earthquake Prediction – Recall Enhancement Using CTGAN</h1>
<p style="color:white;text-align:center;">A Research Pipeline for Rare Event Sensitivity Improvement</p>
</div>
""", unsafe_allow_html=True)

DATA_PATH = "data/query.csv"
MODEL_PKL = "enhanced_rf_model.pkl"

# --------------------------------------------------------------------
# CACHED HELPERS
# --------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw_csv(path):
    return pd.read_csv(path)

@st.cache_resource(show_spinner=False)
def train_ctgan_once(train_df, epochs=300):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=train_df)
    model = CTGANSynthesizer(metadata, epochs=epochs, verbose=False)
    model.fit(train_df)
    return model

@st.cache_resource(show_spinner=False)
def train_random_forest(X, y):
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

# --------------------------------------------------------------------
# START BUTTON
# --------------------------------------------------------------------
if not st.session_state.analysis_done:
    if st.button("🚀 Start Research Analysis"):
        st.session_state.analysis_done = True
        st.rerun()

# --------------------------------------------------------------------
# MAIN PIPELINE (Runs Only Once)
# --------------------------------------------------------------------
if st.session_state.analysis_done:

    # ---------------------------------------------------------
    # PHASE 1 — LOAD & CLEAN DATA
    # ---------------------------------------------------------
    st.header("📍 Phase 1: Preparing Real Earthquake Data")

    with st.spinner("Loading earthquake dataset..."):
        if not os.path.exists(DATA_PATH):
            st.error(f"File {DATA_PATH} not found.")
            st.stop()

        raw_df = load_raw_csv(DATA_PATH)
        real_data = raw_df.copy()

        real_data['time'] = pd.to_datetime(real_data['time'], errors='coerce')

        features = ['latitude', 'longitude', 'depth', 'mag', 'magType', 'nst', 'gap', 'rms']
        real_data = real_data[features].dropna()

        real_data = pd.get_dummies(real_data, columns=['magType'])
        real_data['is_major_quake'] = (real_data['mag'] > 5.5).astype(int)
        real_data.drop("mag", axis=1, inplace=True)

        X = real_data.drop("is_major_quake", axis=1)
        y = real_data["is_major_quake"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        train_data = pd.concat([X_train, y_train], axis=1)

    # ----- CLASS DISTRIBUTION -----
    dist_cols = st.columns([1,2,1])
    with dist_cols[1]:
        fig, ax = plt.subplots(figsize=(5,2))
        real_data['is_major_quake'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title("Class Distribution (Original Data)", fontsize=10)
        st.pyplot(fig)

    # ---------------------------------------------------------
    # PHASE 2 — CTGAN SYNTHETIC DATA
    # ---------------------------------------------------------
    st.header("🧠 Phase 2: Synthetic Data Generation (CTGAN)")
    st.info("Using CTGAN → epochs = 300, batch = 50,000")

    with st.spinner("Training CTGAN (one-time)…"):
        synthesizer = train_ctgan_once(train_data, epochs=300)

    with st.spinner("Generating synthetic major earthquakes…"):
        counts = train_data["is_major_quake"].value_counts()
        minor = counts.get(0, 0)
        major = counts.get(1, 0)
        needed = minor - major

        synthetic = pd.DataFrame()
        batch_size = 50000

        attempts = 0
        while len(synthetic) < needed and attempts < 5:
            attempts += 1
            sample = synthesizer.sample(batch_size)
            sample["is_major_quake"] = pd.to_numeric(sample["is_major_quake"],
                                                     errors="coerce").fillna(0).astype(int)
            synthetic = pd.concat([synthetic, sample[sample['is_major_quake'] == 1]])

        synthetic = synthetic.head(needed)
        augmented_data = pd.concat([train_data, synthetic], ignore_index=True)

    # After CTGAN distribution
    aug_cols = st.columns([1,2,1])
    with aug_cols[1]:
        fig, ax = plt.subplots(figsize=(3,2))
        augmented_data["is_major_quake"].value_counts().plot(kind="bar", ax=ax)
        ax.set_title("After Augmentation (Balanced)", fontsize=10)
        st.pyplot(fig)

    # ---------------------------------------------------------
    # PHASE 3 — MODEL TRAINING + METRICS
    # ---------------------------------------------------------
    st.header("📊 Phase 3: Baseline vs Enhanced Model Evaluation")

    baseline_model = train_random_forest(
        train_data.drop("is_major_quake", axis=1),
        train_data["is_major_quake"]
    )
    pred_base = baseline_model.predict(X_test)
    report_base = classification_report(y_test, pred_base, output_dict=True)

    enhanced_model = train_random_forest(
        augmented_data.drop("is_major_quake", axis=1),
        augmented_data["is_major_quake"]
    )
    pred_enh = enhanced_model.predict(X_test)
    report_enh = classification_report(y_test, pred_enh, output_dict=True)

    st.session_state.training_columns = list(
        augmented_data.drop("is_major_quake", axis=1).columns
    )

    def format_report(r):
        df = pd.DataFrame(r).transpose()
        return df[["precision", "recall", "f1-score", "support"]]

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Baseline Metrics")
        st.dataframe(format_report(report_base))
    with colB:
        st.subheader("Enhanced Metrics")
        st.dataframe(format_report(report_enh))

    # Recall Comparison Chart
    recall_cols = st.columns([1,2,1])
    with recall_cols[1]:
        fig, ax = plt.subplots(figsize=(3,2))
        ax.bar(["Baseline","Enhanced"],
               [report_base["1"]["recall"], report_enh["1"]["recall"]])
        ax.set_title("Major Earthquake Recall Improvement", fontsize=10)
        ax.set_ylim(0,1)
        st.pyplot(fig)

    # Feature Importance
    st.subheader("🧩 Feature Importance")
    imp_cols = st.columns([1,2,1])
    with imp_cols[1]:
        importances = pd.Series(enhanced_model.feature_importances_, index=X_train.columns)
        fig, ax = plt.subplots(figsize=(3,2))
        importances.nlargest(10).plot(kind="barh", ax=ax)
        ax.set_title("Top 10 Features", fontsize=10)
        st.pyplot(fig)

    # ---------------------------------------------------------
    # EARTHQUAKE MAP
    # ---------------------------------------------------------
    st.header("🗺️ Earthquake Map Visualization")

    map_df = raw_df.dropna(subset=["latitude","longitude","mag"])
    eq_map = folium.Map(
        location=[map_df["latitude"].median(), map_df["longitude"].median()],
        zoom_start=2
    )

    for _, r in map_df.iterrows():
        color = "red" if r["mag"] > 5.5 else "orange"
        folium.CircleMarker(
            location=[r["latitude"], r["longitude"]],
            radius=4,
            color=color,
            fill=True,
            popup=f"Mag: {r['mag']}<br>Depth: {r['depth']} km"
        ).add_to(eq_map)

    map_cols = st.columns([1,6,1])
    with map_cols[1]:
        st_folium(eq_map, width=800, height=450)

    # Save Model
    joblib.dump(enhanced_model, MODEL_PKL)
    with open(MODEL_PKL, "rb") as f:
        st.download_button("⬇ Download Enhanced Model", f, file_name=MODEL_PKL)

    st.success("🎉 Research Analysis Completed ✔️")
