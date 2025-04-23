import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="KNN - Prediksi Diabetes", layout="centered")
st.title("🩺 Prediksi Diabetes dengan K-Nearest Neighbors (KNN)")

# Upload dataset
uploaded_file = st.file_uploader("📂 Upload Dataset Diabetes (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Data Sampel")
    st.write(df.head())
    st.write("Kolom:", df.columns.tolist())

    df = df.dropna()  # Bersihkan missing values

    # Encode kolom kategorikal
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    target_col = 'diabetes' if 'diabetes' in df.columns else df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Normalisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Pilih K
    k = st.slider("🔢 Pilih jumlah tetangga (K)", 1, 15, 5)

    # Model
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.success(f"✅ Akurasi Model: {acc:.2f}")

    # Mapping label ramah pengguna (dengan Bahasa Indonesia)
    label_map_display = {
        "gender": {
            0: "0 (Laki-laki)",
            1: "1 (Perempuan)",
            2: "2 (Lainnya)"
        },
        "hypertension": {
            0: "0 (Tidak)",
            1: "1 (Ya)"
        },
        "heart_disease": {
            0: "0 (Tidak)",
            1: "1 (Ya)"
        },
        "smoking_history": {
            0: "0 (Tidak ada info)",
            1: "1 (Tidak Pernah)",
            2: "2 (Dulu Pernah)",
            3: "3 (Kadang-kadang)",
            4: "4 (Masih Merokok)",
            5: "5 (Pernah Merokok)"
        }
    }

    # Reverse mapping untuk diproses
    label_map_reverse = {
        col: {v: k for k, v in label.items()}
        for col, label in label_map_display.items()
    }

    st.subheader("🧍 Prediksi Pasien Baru")
    with st.form("form_input"):
        input_data = {}
        for col in X.columns:
            if col in label_map_display:
                options = list(label_map_display[col].values())
                selected = st.selectbox(f"{col} (dalam kurung: Bahasa Indonesia)", options)
                val = label_map_reverse[col][selected]
            elif df[col].nunique() <= 10:
                val = st.selectbox(f"{col}", sorted(df[col].unique()))
            else:
                val = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
            input_data[col] = val
        submitted = st.form_submit_button("🔍 Prediksi")

    if submitted:
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        result = model.predict(input_scaled)[0]

        if result == 1:
            st.error("⚠️ Pasien kemungkinan MENGIDAP diabetes.")
        else:
            st.success("✅ Pasien kemungkinan TIDAK mengidap diabetes.")
else:
    st.info("Silakan upload dataset diabetes (.csv) terlebih dahulu.")
