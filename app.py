import streamlit as st
import pandas as pd
import numpy as np
import joblib
import difflib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

st.set_page_config(
    page_title="CarPrice.id — Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Base */
[data-testid="stAppViewContainer"] { background: #0f1117; color: #e0e0e0; }
[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #1e2535;
}
[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; color: #a0aec0; }
[data-testid="stSidebar"] .stRadio label:hover { color: #63b3ed; }
section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

/* Header brand */
.brand-header {
    display: flex; align-items: center; gap: 14px;
    padding: 1rem 0 1.5rem 0;
    border-bottom: 1px solid #1e2535;
    margin-bottom: 1.5rem;
}
.brand-icon { font-size: 2.4rem; }
.brand-title { font-size: 1.5rem; font-weight: 700; color: #fff; line-height: 1.1; }
.brand-subtitle { font-size: 0.78rem; color: #63b3ed; letter-spacing: 0.06em; text-transform: uppercase; }

/* Page title */
.page-title {
    font-size: 1.7rem; font-weight: 700; color: #fff;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #2d6eea;
    margin-bottom: 1.5rem;
    display: inline-block;
}

/* Metric cards */
.metric-card {
    background: #1a2035;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.metric-label { font-size: 0.78rem; color: #718096; text-transform: uppercase; letter-spacing: 0.07em; }
.metric-value { font-size: 1.8rem; font-weight: 700; color: #63b3ed; margin: 0.2rem 0 0 0; }

/* Info box */
.info-box {
    background: #1a2035;
    border-left: 4px solid #2d6eea;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    font-size: 0.9rem;
    color: #cbd5e0;
}

/* Prediction result */
.result-card {
    background: linear-gradient(135deg, #1a3a5c 0%, #0f2444 100%);
    border: 1px solid #2d6eea;
    border-radius: 14px;
    padding: 1.8rem 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-label { font-size: 0.85rem; color: #90cdf4; text-transform: uppercase; letter-spacing: 0.1em; }
.result-price { font-size: 2.8rem; font-weight: 800; color: #fff; margin: 0.4rem 0 0 0; }

/* Step pills */
.step-pill {
    display: inline-block;
    background: #1e2d4a;
    color: #63b3ed;
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* Divider */
hr { border-color: #1e2535 !important; }

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #2d6eea, #1a56db);
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 2rem;
    font-weight: 600;
    font-size: 0.95rem;
    transition: opacity 0.2s;
    width: 100%;
}
.stButton > button:hover { opacity: 0.88; }

/* Input labels */
.stTextInput label, .stSelectbox label, .stNumberInput label {
    color: #a0aec0 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* Alert overrides */
.stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Shared chart theme ──────────────────────────────────────────────────────────
CHART_BG   = "#151b2d"
AXIS_COLOR = "#374151"
TEXT_COLOR = "#9ca3af"
ACCENT     = "#3b82f6"

def apply_chart_style(fig, ax):
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color("#e2e8f0")
    for spine in ax.spines.values():
        spine.set_edgecolor(AXIS_COLOR)
    ax.grid(color=AXIS_COLOR, linestyle="--", linewidth=0.5, alpha=0.5)

# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data
def load_raw_data():
    df1 = pd.read_csv("dataset/used_car.csv")
    df2 = pd.read_csv("dataset/used_car_data_new.csv")
    return df1, df2

@st.cache_data
def load_processed_data():
    df1, df2 = load_raw_data()
    df2['transmission'] = df2['id_transmission'].map({1: 'manual', 2: 'automatic'})
    df1['mileage (km)'] = df1['mileage (km)'] * 1000
    df = pd.concat([df1, df2], ignore_index=True)
    df = df[['brand', 'model', 'year', 'mileage (km)', 'transmission', 'price (Rp)']]
    df['brand']        = df['brand'].astype(str).str.lower().str.strip()
    df['model']        = df['model'].astype(str).str.lower().str.strip()
    df['transmission'] = df['transmission'].astype(str).str.lower().str.strip()
    return df

df1_raw, df2_raw = load_raw_data()
df               = load_processed_data()

# ── Model ──────────────────────────────────────────────────────────────────────
model        = joblib.load("model/model.pkl")
le_trans     = joblib.load("model/le_trans.pkl")
model_columns = joblib.load("model/model_columns.pkl")

# ── Smart match ────────────────────────────────────────────────────────────────
def smart_match(input_str, choices, threshold=0.5):
    input_str = input_str.lower().strip()
    if input_str in choices:
        return input_str, 1.0
    best_match, best_score = None, 0
    for c in choices:
        score = difflib.SequenceMatcher(None, input_str, c).ratio()
        if score > best_score:
            best_score, best_match = score, c
    return (best_match, best_score) if best_score >= threshold else (None, best_score)

valid_brands    = df['brand'].unique()
valid_models    = df['model'].unique()
brand_model_map = df.groupby('brand')['model'].unique().to_dict()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand-header">
        <div class="brand-icon">🚗</div>
        <div>
            <div class="brand-title">CarPrice<span style="color:#3b82f6">.id</span></div>
            <div class="brand-subtitle">ML Price Predictor</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "Navigation",
        ["📂  Dataset", "📊  EDA", "⚙️  Preprocessing", "🧠  Training", "🔮  Prediction"],
        label_visibility="collapsed",
    )
    menu = menu.split("  ", 1)[1]   # strip icon prefix

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.75rem;color:#4a5568;text-align:center;'>Group 02</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:0.75rem;color:#4a5568;text-align:center;'>Justin C.K., Brian N.T., Marvin A.R.,</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:0.75rem;color:#4a5568;text-align:center;'>Jason B., Kian A.W.</div>",
        unsafe_allow_html=True,
    )

# ── Dataset ────────────────────────────────────────────────────────────────────
if menu == "Dataset":
    st.markdown('<div class="page-title">📂 Raw Dataset</div>', unsafe_allow_html=True)

    t1, t2 = st.tabs(["Dataset 1 — used_car.csv", "Dataset 2 — used_car_data_new.csv"])

    with t1:
        col_a, col_b, col_c = st.columns(3)
        col_a.markdown(f'<div class="metric-card"><div class="metric-label">Rows</div><div class="metric-value">{df1_raw.shape[0]:,}</div></div>', unsafe_allow_html=True)
        col_b.markdown(f'<div class="metric-card"><div class="metric-label">Columns</div><div class="metric-value">{df1_raw.shape[1]}</div></div>', unsafe_allow_html=True)
        col_c.markdown(f'<div class="metric-card"><div class="metric-label">Missing Values</div><div class="metric-value">{df1_raw.isnull().sum().sum()}</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df1_raw.head(500), use_container_width=True)

    with t2:
        col_a, col_b, col_c = st.columns(3)
        col_a.markdown(f'<div class="metric-card"><div class="metric-label">Rows</div><div class="metric-value">{df2_raw.shape[0]:,}</div></div>', unsafe_allow_html=True)
        col_b.markdown(f'<div class="metric-card"><div class="metric-label">Columns</div><div class="metric-value">{df2_raw.shape[1]}</div></div>', unsafe_allow_html=True)
        col_c.markdown(f'<div class="metric-card"><div class="metric-label">Missing Values</div><div class="metric-value">{df2_raw.isnull().sum().sum()}</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df2_raw.head(500), use_container_width=True)

# ── EDA ────────────────────────────────────────────────────────────────────────
elif menu == "EDA":
    st.markdown('<div class="page-title">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    # Overview metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-label">Total Records</div><div class="metric-value">{df.shape[0]:,}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-label">Features</div><div class="metric-value">{df.shape[1]}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-label">Unique Brands</div><div class="metric-value">{df["brand"].nunique()}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-label">Missing Values</div><div class="metric-value">{df.isnull().sum().sum()}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("Descriptive Statistics", expanded=False):
        st.dataframe(df.describe(), use_container_width=True)

    st.markdown("---")

    # Row 1: Price distribution + Top brand
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Price Distribution**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.histplot(df['price (Rp)'], bins=40, kde=True, ax=ax, color=ACCENT, alpha=0.7)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rp {x/1e6:.0f}M"))
        ax.set_xlabel("Price (Rp)")
        ax.set_ylabel("Count")
        ax.set_title("Car Price Distribution")
        apply_chart_style(fig, ax)
        st.pyplot(fig, use_container_width=True)

    with col_r:
        st.markdown("**Top 10 Brands**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        top_brands = df['brand'].value_counts().head(10)
        colors = [ACCENT if i == 0 else "#2d3f6e" for i in range(len(top_brands))]
        top_brands.plot(kind='barh', ax=ax, color=colors[::-1])
        ax.invert_yaxis()
        ax.set_xlabel("Count")
        ax.set_title("Top 10 Brands by Listing")
        apply_chart_style(fig, ax)
        st.pyplot(fig, use_container_width=True)

    # Row 2: Transmission + Year
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Transmission Type**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        trans_counts = df['transmission'].value_counts()
        ax.pie(
            trans_counts,
            labels=[t.title() for t in trans_counts.index],
            autopct="%1.1f%%",
            colors=["#3b82f6", "#1e40af"],
            startangle=90,
            textprops={"color": "#e2e8f0", "fontsize": 10},
        )
        ax.set_title("Manual vs Automatic")
        fig.patch.set_facecolor(CHART_BG)
        ax.set_facecolor(CHART_BG)
        ax.title.set_color("#e2e8f0")
        st.pyplot(fig, use_container_width=True)

    with col_r:
        st.markdown("**Car Year Distribution**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.histplot(df['year'], bins=20, ax=ax, color="#60a5fa", alpha=0.8)
        ax.set_xlabel("Year")
        ax.set_ylabel("Count")
        ax.set_title("Manufacturing Year Spread")
        apply_chart_style(fig, ax)
        st.pyplot(fig, use_container_width=True)

    # Row 3: Scatter plots
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Mileage vs Price**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.scatter(df['mileage (km)'], df['price (Rp)'], alpha=0.25, s=8, color=ACCENT)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rp {x/1e6:.0f}M"))
        ax.set_xlabel("Mileage (km)")
        ax.set_ylabel("Price (Rp)")
        ax.set_title("Mileage vs Price")
        apply_chart_style(fig, ax)
        st.pyplot(fig, use_container_width=True)

    with col_r:
        st.markdown("**Year vs Price**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.scatter(df['year'], df['price (Rp)'], alpha=0.25, s=8, color="#a78bfa")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rp {x/1e6:.0f}M"))
        ax.set_xlabel("Year")
        ax.set_ylabel("Price (Rp)")
        ax.set_title("Year vs Price")
        apply_chart_style(fig, ax)
        st.pyplot(fig, use_container_width=True)

# ── Preprocessing ──────────────────────────────────────────────────────────────
elif menu == "Preprocessing":
    st.markdown('<div class="page-title">⚙️ Preprocessing</div>', unsafe_allow_html=True)

    df_clean = df.dropna().drop_duplicates()

    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card"><div class="metric-label">Before Cleaning</div><div class="metric-value">{df.shape[0]:,}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-label">After Cleaning</div><div class="metric-value">{df_clean.shape[0]:,}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-label">Rows Removed</div><div class="metric-value">{df.shape[0] - df_clean.shape[0]:,}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Steps applied: <b>drop NaN rows</b> → <b>drop exact duplicates</b> → <b>lowercase & trim text columns</b> (brand, model, transmission)
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(df_clean.head(800), use_container_width=True)

# ── Training ───────────────────────────────────────────────────────────────────
elif menu == "Training":
    st.markdown('<div class="page-title">🧠 Model Training & Evaluation</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("#### Algorithm")
        st.markdown("""
        <div class="info-box">
        <b>Random Forest Regressor</b><br><br>
        An ensemble of decision trees that reduces overfitting by averaging predictions and handles non-linear patterns inherent in used-car pricing.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Training Pipeline")
        steps = [
            ("1", "Data Preparation", "Merged two datasets, removed nulls & duplicates, normalised text"),
            ("2", "Feature Engineering", "Derived `car_age = 2026 − year`"),
            ("3", "Encoding", "Label Encoding → transmission · One-Hot Encoding → brand & model"),
            ("4", "Target Transform", "Applied `log1p(price)` to reduce skewness"),
            ("5", "Data Split", "80 % training · 20 % testing"),
            ("6", "Model Fit", "RandomForestRegressor(n_estimators=150)"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div style="display:flex;gap:12px;align-items:flex-start;margin-bottom:0.85rem;">
                <div style="min-width:28px;height:28px;border-radius:50%;background:#1e3a5f;color:#63b3ed;
                            display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.8rem;">
                    {num}
                </div>
                <div>
                    <div style="font-weight:600;color:#e2e8f0;font-size:0.9rem;">{title}</div>
                    <div style="color:#718096;font-size:0.82rem;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with right:
        st.markdown("#### Model Performance")
        m1, m2, m3 = st.columns(3)
        m1.markdown('<div class="metric-card"><div class="metric-label">MAE</div><div class="metric-value">Rp 22.9M</div></div>', unsafe_allow_html=True)
        m2.markdown('<div class="metric-card"><div class="metric-label">RMSE</div><div class="metric-value">Rp 35.8M</div></div>', unsafe_allow_html=True)
        m3.markdown('<div class="metric-card"><div class="metric-label">R² Score</div><div class="metric-value">0.59</div></div>', unsafe_allow_html=True)
        st.caption("Evaluated on test set after inverse log transformation.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Top 10 Feature Importance")

        try:
            fi_df = pd.DataFrame({
                "feature":    model_columns,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(6, 3.8))
            colors = [ACCENT if i == 0 else "#1e3a5f" for i in range(len(fi_df))]
            ax.barh(fi_df["feature"], fi_df["importance"], color=colors[::-1])
            ax.invert_yaxis()
            ax.set_xlabel("Importance Score")
            ax.set_title("Top 10 Feature Importances")
            apply_chart_style(fig, ax)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        except Exception:
            st.warning("Feature importance not available.")

    st.markdown("---")
    st.success("**Conclusion:** Mileage and car age are the strongest predictors of price. Brand and model also carry significant weight.")

# ── Prediction ─────────────────────────────────────────────────────────────────
elif menu == "Prediction":
    st.markdown('<div class="page-title">🔮 Price Prediction</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Enter your car details below. Brand and model names are matched with fuzzy logic — typos are handled automatically.
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state for mileage synchronization
    if 'mileage' not in st.session_state:
        st.session_state.mileage = 50_000

    st.markdown("**Mileage (km)**")
    m1, m2 = st.columns([2, 1])
    with m1:
        st.session_state.mileage = st.slider("Slider", min_value=0, max_value=500_000, 
                                             value=st.session_state.mileage, step=500,
                                             format="%d km", label_visibility="collapsed")
    with m2:
        st.session_state.mileage = st.number_input("Manual Input", min_value=0, max_value=500_000, 
                                                   value=st.session_state.mileage, step=100,
                                                   label_visibility="collapsed")

    mileage = st.session_state.mileage

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            brand_input = st.text_input("Car Brand", placeholder="e.g. Toyota, Honda, Daihatsu")
            model_input = st.text_input("Car Model", placeholder="e.g. Innova, Brio, Agya")
        with c2:
            transmission = st.selectbox("Transmission", options=le_trans.classes_)
            year         = st.number_input("Manufacturing Year", min_value=2000, max_value=2026, value=2020)

        submitted = st.form_submit_button("🔍  Predict Price")

    if submitted:
        if not brand_input.strip() or not model_input.strip():
            st.error("Please fill in both Brand and Model fields.")
            st.stop()

        with st.spinner("Running prediction…"):
            brand_match, _ = smart_match(brand_input, valid_brands)
            if brand_match is None:
                st.error(f"Brand **{brand_input}** not found in training data.")
                st.stop()

            filtered_models = brand_model_map.get(brand_match, valid_models)
            model_match, _  = smart_match(model_input, filtered_models)
            if model_match is None:
                st.error(f"Model **{model_input}** not found for brand **{brand_match}**.")
                st.stop()

            # Interpretation summary
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.markdown(f'<div class="metric-card"><div class="metric-label">Brand</div><div class="metric-value" style="font-size:1.2rem;">{brand_match.title()}</div></div>', unsafe_allow_html=True)
            col_b.markdown(f'<div class="metric-card"><div class="metric-label">Model</div><div class="metric-value" style="font-size:1.2rem;">{model_match.title()}</div></div>', unsafe_allow_html=True)
            col_c.markdown(f'<div class="metric-card"><div class="metric-label">Year</div><div class="metric-value" style="font-size:1.2rem;">{year}</div></div>', unsafe_allow_html=True)
            col_d.markdown(f'<div class="metric-card"><div class="metric-label">Mileage</div><div class="metric-value" style="font-size:1.2rem;">{mileage:,} km</div></div>', unsafe_allow_html=True)

            # Feature engineering
            car_age  = 2026 - year
            input_df = pd.DataFrame(0, index=[0], columns=model_columns)
            input_df['mileage (km)']  = mileage
            input_df['car_age']       = car_age
            input_df['transmission']  = le_trans.transform([transmission])[0]

            brand_col = f"brand_{brand_match}"
            if brand_col in model_columns:
                input_df[brand_col] = 1
            else:
                st.error("Brand not recognised by the trained model.")
                st.stop()

            model_col = f"model_{model_match}"
            if model_col in model_columns:
                input_df[model_col] = 1
            else:
                model_candidates = [c.replace("model_", "") for c in model_columns if c.startswith("model_")]
                closest = difflib.get_close_matches(model_match, model_candidates, n=1, cutoff=0.3)
                if closest:
                    input_df[f"model_{closest[0]}"] = 1
                    st.info(f"Model mapped to nearest training label: **{closest[0].title()}**")
                else:
                    st.error("No similar model found in training data.")
                    st.stop()

            pred_log = np.clip(model.predict(input_df)[0], 0, 20)
            pred     = np.expm1(pred_log)

            if np.isinf(pred) or np.isnan(pred):
                st.error("Prediction produced an invalid result. Please check your inputs.")
            else:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Estimated Market Price</div>
                    <div class="result-price">Rp {int(pred):,}</div>
                    <div style="color:#90cdf4;font-size:0.8rem;margin-top:0.6rem;">
                        Based on {brand_match.title()} {model_match.title()} · {year} · {mileage:,} km · {transmission.title()}
                    </div>
                </div>
                """, unsafe_allow_html=True)
