"""
Dashboard — Airline Passenger Satisfaction Classification
===========================================================
Versi DEPLOY: dataset sudah tertanam di folder `data/` (train.csv.gz, test.csv.gz),
sehingga dosen/penilai tinggal membuka link dashboard tanpa perlu upload file apa pun.

Cara deploy (ringkas — lihat README.md untuk detail lengkap):
    1. Push folder ini (dashboard.py, requirements.txt, data/) ke repo GitHub.
    2. Buka https://share.streamlit.io -> New app -> pilih repo -> file: dashboard.py -> Deploy.

Cara jalan lokal:
    pip install -r requirements.txt
    streamlit run dashboard.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

RANDOM_STATE = 42
sns.set_style("whitegrid")

st.set_page_config(
    page_title="Airline Passenger Satisfaction — Dashboard",
    page_icon="✈️",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "train.csv.gz")
TEST_PATH = os.path.join(BASE_DIR, "data", "test.csv.gz")

SERVICE_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Inflight service", "Cleanliness",
]
CATEGORICAL_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]


# ----------------------------------------------------------------------------
# 1. DATA LOADING (otomatis dari folder data/, tidak perlu upload)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Memuat & menggabungkan data...")
def load_data():
    train_raw = pd.read_csv(TRAIN_PATH, compression="gzip")
    test_raw = pd.read_csv(TEST_PATH, compression="gzip")
    df = pd.concat([train_raw, test_raw], ignore_index=True)
    return df


# ----------------------------------------------------------------------------
# 2. CLEANING + FEATURE ENGINEERING + ENCODING (identik dengan notebook)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Menjalankan cleaning, feature engineering, dan encoding...")
def preprocess(df):
    df_clean = df.copy()
    cols_to_drop = [c for c in ["Unnamed: 0", "id"] if c in df_clean.columns]
    df_clean = df_clean.drop(columns=cols_to_drop)

    median_arrival_delay = df_clean["Arrival Delay in Minutes"].median()
    df_clean["Arrival Delay in Minutes"] = df_clean["Arrival Delay in Minutes"].fillna(median_arrival_delay)
    df_clean = df_clean.drop_duplicates()

    df_features = df_clean.copy()
    df_features["avg_service_rating"] = df_features[SERVICE_COLS].mean(axis=1)
    df_features["total_delay"] = (
        df_features["Departure Delay in Minutes"] + df_features["Arrival Delay in Minutes"]
    )

    df_encoded = pd.get_dummies(df_features, columns=CATEGORICAL_COLS, drop_first=True)
    target_encoder = LabelEncoder()
    df_encoded["satisfaction"] = target_encoder.fit_transform(df_encoded["satisfaction"])
    target_mapping = dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))

    target_col = "satisfaction"
    categorical_dummy_cols = [c for c in df_encoded.columns if any(
        c.startswith(prefix + "_") for prefix in CATEGORICAL_COLS
    )]
    numeric_feature_cols = [
        c for c in df_encoded.columns if c not in categorical_dummy_cols and c != target_col
    ]
    target_corr = df_encoded[numeric_feature_cols + [target_col]].corr()[target_col].drop(target_col)
    low_relevance_cols = target_corr[target_corr.abs() < 0.02].index.tolist()

    corr_matrix = df_encoded[numeric_feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    redundant_cols = [c for c in upper.columns if any(upper[c] > 0.9)]

    cols_to_drop_selection = list(set(low_relevance_cols + redundant_cols))
    df_encoded = df_encoded.drop(columns=cols_to_drop_selection)

    numeric_cols_to_scale = [
        c for c in ["Age", "Flight Distance", "Departure Delay in Minutes",
                     "Arrival Delay in Minutes", "total_delay", "avg_service_rating"]
        if c in df_encoded.columns
    ]
    scaler = StandardScaler()
    df_encoded[numeric_cols_to_scale] = scaler.fit_transform(df_encoded[numeric_cols_to_scale])

    return df_clean, df_features, df_encoded, target_mapping, cols_to_drop_selection


# ----------------------------------------------------------------------------
# 3. MODELING
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Melatih model klasifikasi (bisa memakan waktu 1-2 menit pertama kali)...")
def train_models(df_encoded):
    X = df_encoded.drop(columns=["satisfaction"])
    y = df_encoded["satisfaction"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, eval_metric="logloss",
        )

    results = []
    predictions = {}
    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        results.append({
            "model": name,
            "train_accuracy": accuracy_score(y_train, y_pred_train),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
        })
        predictions[name] = (y_test, y_pred)
        fitted_models[name] = model

    results_df = pd.DataFrame(results).set_index("model").round(4)
    return results_df, predictions, fitted_models, X_train, X_test, y_train, y_test


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("Dashboard Airline Passenger Satisfaction")
st.caption(
    "Dashboard analisis kepuasan penumpang maskapai penerbangan"
    "tinggal ditelusuri melalui tab di bawah ini."
)

if not XGBOOST_AVAILABLE:
    st.warning(
        "Library XGBoost tidak terdeteksi di environment ini, sehingga hanya Logistic Regression "
        "dan Random Forest yang ditampilkan. Pastikan `xgboost` tercantum di requirements.txt saat deploy."
    )

df = load_data()
df_clean, df_features, df_encoded, target_mapping, dropped_cols = preprocess(df)
results_df, predictions, fitted_models, X_train, X_test, y_train, y_test = train_models(df_encoded)
best_model_name = results_df["f1_score"].idxmax()

tabs = st.tabs([
    "📋 Ringkasan Data",
    "🔍 EDA",
    "⚙️ Preprocessing",
    "🤖 Performa Model",
    "🎯 Feature Importance",
    "💡 Insight Bisnis",
])

# ----------------------------------------------------------------------------
# TAB 1 — RINGKASAN DATA
# ----------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Ringkasan Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Baris", f"{df.shape[0]:,}")
    c2.metric("Total Kolom (mentah)", df.shape[1])
    c3.metric("Baris setelah cleaning", f"{df_clean.shape[0]:,}")
    c4.metric("Fitur final (setelah seleksi)", df_encoded.shape[1] - 1)

    st.markdown("**Mapping label target (`satisfaction`)**")
    st.json({k: int(v) for k, v in target_mapping.items()})

    st.markdown("**Contoh data mentah**")
    st.dataframe(df.head(10), use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 2 — EDA
# ----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Distribusi Kelas Target**")
        fig, ax = plt.subplots(figsize=(5, 4))
        df_clean["satisfaction"].value_counts().plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
        ax.set_xlabel("Kelas")
        ax.set_ylabel("Jumlah Penumpang")
        plt.xticks(rotation=20)
        st.pyplot(fig)

    with col2:
        st.markdown("**Distribusi Umur berdasarkan Kelas Penerbangan & Kepuasan**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(data=df_clean, x="Class", y="Age", hue="satisfaction", ax=ax)
        st.pyplot(fig)

    st.markdown("**Rata-rata Rating Layanan berdasarkan Status Kepuasan**")
    rating_by_satisfaction = df_clean.groupby("satisfaction")[SERVICE_COLS].mean().T
    fig, ax = plt.subplots(figsize=(9, 7))
    rating_by_satisfaction.plot(kind="barh", ax=ax)
    ax.set_xlabel("Rata-rata Rating")
    st.pyplot(fig)

    st.markdown("**Korelasi Antar Fitur Numerik**")
    fig, ax = plt.subplots(figsize=(9, 7))
    correlation_matrix = df_clean[SERVICE_COLS + ["Age", "Flight Distance"]].corr()
    sns.heatmap(correlation_matrix, cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)

# ----------------------------------------------------------------------------
# TAB 3 — PREPROCESSING
# ----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Ringkasan Tahapan Preprocessing")
    st.markdown(
        """
| Tahap | Detail | Alasan |
|---|---|---|
| Missing value handling | `Arrival Delay in Minutes` diisi median | Distribusi delay memiliki outlier ekstrem (long tail) |
| Duplicate removal | `drop_duplicates()` | Menghindari bias pembelajaran akibat baris identik |
| Feature engineering | `avg_service_rating`, `total_delay` | Meringkas 14 kolom rating & 2 kolom delay jadi indikator komposit |
| Encoding | One-hot (`Gender`, `Customer Type`, `Type of Travel`, `Class`) + Label Encoding target | Model klasifikasi butuh input numerik |
| Feature selection | Filter relevansi (\\|corr\\| < 0.02) + multikolinearitas (>0.9) | Mengurangi redundansi & fitur tidak informatif |
| Standardisasi | `StandardScaler` pada kolom numerik | Menyamakan skala antar fitur numerik |
| Train-test split | 80:20, stratifikasi pada target | Menjaga proporsi kelas pada data uji |
        """
    )
    st.markdown(f"**Fitur yang dibuang pada tahap feature selection ({len(dropped_cols)}):**")
    st.write(dropped_cols if dropped_cols else "Tidak ada fitur yang dibuang.")

# ----------------------------------------------------------------------------
# TAB 4 — PERFORMA MODEL
# ----------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Perbandingan Algoritma Klasifikasi")
    st.dataframe(results_df.style.highlight_max(axis=0, color="#c6efce"), use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    results_df.plot(kind="bar", ax=ax, ylim=(0.7, 1.0))
    plt.xticks(rotation=0)
    plt.ylabel("Skor")
    st.pyplot(fig)

    st.success(f"🏆 Model dengan F1-score tertinggi: **{best_model_name}**")

    st.markdown("---")
    st.markdown("### Detail per Model")
    selected_model = st.selectbox("Pilih model untuk melihat detail", list(fitted_models.keys()))

    y_true, y_pred = predictions[selected_model]
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Parameter Model**")
        st.json({k: str(v) for k, v in fitted_models[selected_model].get_params().items()})

    with col2:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["neutral/dissatisfied", "satisfied"],
                    yticklabels=["neutral/dissatisfied", "satisfied"], ax=ax)
        ax.set_ylabel("Aktual")
        ax.set_xlabel("Prediksi")
        st.pyplot(fig)

    st.markdown("**Classification Report**")
    report = classification_report(
        y_true, y_pred, target_names=["neutral or dissatisfied", "satisfied"], output_dict=True
    )
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 5 — FEATURE IMPORTANCE
# ----------------------------------------------------------------------------
with tabs[4]:
    st.subheader(f"Feature Importance Model Terbaik ({best_model_name})")

    best_model = fitted_models[best_model_name]
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=X_train.columns)
    else:
        importances = pd.Series(np.abs(best_model.coef_[0]), index=X_train.columns)

    top_features = importances.sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    top_features.sort_values().plot(kind="barh", ax=ax)
    ax.set_xlabel("Importance Score")
    st.pyplot(fig)

    st.dataframe(top_features.reset_index().rename(columns={"index": "Fitur", 0: "Importance"}),
                 use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 6 — INSIGHT BISNIS
# ----------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Insight Bisnis")

    best_row = results_df.loc[best_model_name]
    st.markdown(
        f"""
**Temuan Utama (otomatis dari hasil di atas)**
- Model terbaik: **{best_model_name}** dengan F1-score **{best_row['f1_score']:.3f}**,
  akurasi testing **{best_row['accuracy']:.3f}**.
- 5 fitur paling berpengaruh terhadap kepuasan penumpang:
  {", ".join(top_features.sort_values(ascending=False).head(5).index.tolist())}.
        """
    )

    st.markdown(
        """
**Insight bisnis, rekomendasi, keterbatasan, dan saran pengembangan versi lengkap**
dijelaskan secara naratif pada dokumen laporan (Word) yang menyertai dashboard ini.
Bagian ini menampilkan ringkasan otomatis berbasis angka aktual hasil run terkini.
        """
    )

st.markdown("---")
st.caption("Dashboard dibuat dengan Streamlit")
