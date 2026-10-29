import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Simulasi Risiko Klaim Asuransi", layout="wide")
st.title("Simulasi Monte Carlo: Risiko Klaim Asuransi")
st.write(
    "Estimasi probabilitas **total klaim N pasien baru melebihi anggaran perusahaan**, "
    "berdasarkan data *Medical Cost Personal Dataset* (kolom `charges`)."
)

# ---------- Sidebar ----------
st.sidebar.header("Pengaturan")
upload = st.sidebar.file_uploader("Upload insurance.csv (opsional)", type="csv")
n_patients = st.sidebar.number_input("Jumlah pasien baru", 10, 1000, 100, step=10)
n_sim = st.sidebar.number_input("Jumlah skenario simulasi", 5000, 200000, 10000, step=5000)
budget_factor = st.sidebar.slider("Anggaran (x ekspektasi total klaim)", 0.90, 1.50, 1.05, 0.01)
metode = st.sidebar.selectbox("Metode pembangkitan", ["Normal", "Lognormal", "Bootstrap"])
remove_outlier = st.sidebar.checkbox("Buang outlier ekstrem (3 x IQR)", value=True)
seed = st.sidebar.number_input("Seed (isi NIM)", 0, 2**32 - 1, 0)


@st.cache_data
def load_data(file):
    return pd.read_csv(file if file is not None else "insurance.csv")


try:
    df = load_data(upload)
except FileNotFoundError:
    st.error("File insurance.csv tidak ditemukan. Upload lewat sidebar atau taruh di folder yang sama dengan app.py.")
    st.stop()

df = df.dropna().drop_duplicates()
if remove_outlier:
    q1, q3 = df["charges"].quantile([0.25, 0.75])
    iqr = q3 - q1
    df = df[(df["charges"] >= q1 - 3 * iqr) & (df["charges"] <= q3 + 3 * iqr)]

charges = df["charges"].values
mu, sigma = charges.mean(), charges.std(ddof=1)

# ---------- Statistik data ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Jumlah data", f"{len(charges):,}")
c2.metric("Mean charges", f"{mu:,.0f}")
c3.metric("Std Dev", f"{sigma:,.0f}")
c4.metric("Skewness", f"{pd.Series(charges).skew():.2f}")

with st.expander("Histogram data asli"):
    fig0, ax0 = plt.subplots(figsize=(7, 3))
    ax0.hist(charges, bins=40, color="steelblue", edgecolor="white")
    ax0.set_xlabel("charges")
    ax0.set_ylabel("Frekuensi")
    st.pyplot(fig0)

# ---------- Simulasi ----------
if st.button("Jalankan Simulasi", type="primary"):
    rng = np.random.default_rng(int(seed))
    n_sim, n_patients = int(n_sim), int(n_patients)

    if metode == "Normal":
        total = np.clip(rng.normal(mu, sigma, (n_sim, n_patients)), 0, None).sum(axis=1)
    elif metode == "Lognormal":
        lc = np.log(charges)
        total = rng.lognormal(lc.mean(), lc.std(ddof=1), (n_sim, n_patients)).sum(axis=1)
    else:
        total = charges[rng.integers(0, len(charges), (n_sim, n_patients))].sum(axis=1)

    budget = budget_factor * n_patients * mu
    prob = np.mean(total > budget)

    st.subheader("Hasil")
    r1, r2, r3 = st.columns(3)
    r1.metric("Anggaran", f"{budget:,.0f}")
    r2.metric("Rata-rata total klaim", f"{total.mean():,.0f}")
    r3.metric("P(klaim > anggaran)", f"{prob:.2%}")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(total, bins=50, color="steelblue", edgecolor="white")
    ax.axvline(budget, color="red", linestyle="--", linewidth=2, label=f"Anggaran = {budget:,.0f}")
    ax.axvspan(budget, total.max(), color="red", alpha=0.12, label="Area risiko")
    ax.set_xlabel(f"Total klaim {n_patients} pasien")
    ax.set_ylabel("Frekuensi skenario")
    ax.set_title(f"Distribusi total klaim ({metode}, {n_sim:,} skenario)")
    ax.legend()
    st.pyplot(fig)

    st.caption(
        f"Dari {n_sim:,} skenario, {int((total > budget).sum()):,} skenario melebihi anggaran. "
        f"Persentil 95 total klaim: {np.percentile(total, 95):,.0f}."
    )
else:
    st.info("Atur parameter di sidebar, lalu klik **Jalankan Simulasi**.")
