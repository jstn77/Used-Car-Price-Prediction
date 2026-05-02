import streamlit as st
import pandas as pd
import numpy as np
import joblib
import difflib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import seaborn as sns

st.set_page_config(
    page_title="CarPrice.id — Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
with open("styles.css") as _f:
    st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

# ── Theme-aware chart palette ──────────────────────────────────────────────────
try:
    _base = st.get_option("theme.base") or "dark"
except Exception:
    _base = "dark"

LIGHT = (_base == "light")

CHART_BG   = "#f4f4f5" if LIGHT else "#18181b"
AXIS_COLOR = "#e4e4e7" if LIGHT else "#27272a"
TEXT_COLOR = "#71717a"
FG_COLOR   = "#09090b" if LIGHT else "#fafafa"
ACCENT     = "#2563eb" if LIGHT else "#3b82f6"
ACCENT2    = "#8b5cf6"
AMBER      = "#f59e0b"
EMERALD    = "#10b981"

plt.rcParams.update({
    "font.family"      : "sans-serif",
    "font.size"        : 9,
    "axes.titlesize"   : 10,
    "axes.labelsize"   : 9,
    "xtick.labelsize"  : 8,
    "ytick.labelsize"  : 8,
    "figure.dpi"       : 130,
})

def styled_fig(w=7, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)
    return fig, ax

def apply_chart_style(fig, ax):
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(FG_COLOR)
    ax.title.set_fontweight("600")
    for spine in ax.spines.values():
        spine.set_edgecolor(AXIS_COLOR)
    ax.grid(color=AXIS_COLOR, linestyle="--", linewidth=0.4, alpha=0.6)
    fig.tight_layout()

def bar_colors(n, base=ACCENT, dim=None):
    dim = dim or AXIS_COLOR
    r0, g0, b0 = mcolors.to_rgb(dim)
    r1, g1, b1 = mcolors.to_rgb(base)
    return [(r0 + (r1-r0)*t, g0 + (g1-g0)*t, b0 + (b1-b0)*t)
            for t in np.linspace(0.35, 1.0, n)]

def add_bar_labels(ax, bars, fmt="{:.0f}", color=None):
    color = color or TEXT_COLOR
    max_v = max(b.get_width() for b in bars) if hasattr(bars[0], "get_width") else \
            max(b.get_height() for b in bars)
    for bar in bars:
        is_h = hasattr(bar, "get_width") and bar.get_width() > 0
        if is_h:
            v = bar.get_width()
            ax.text(v + max_v * 0.015, bar.get_y() + bar.get_height() / 2,
                    fmt.format(v), va="center", ha="left", fontsize=7.5, color=color)
        else:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, v + max_v * 0.01,
                    fmt.format(v), va="bottom", ha="center", fontsize=7.5, color=color)

# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data
def load_raw_data():
    df1 = pd.read_csv("dataset/used_car.csv")
    df2 = pd.read_csv("dataset/used_car_data_new.csv")
    return df1, df2

@st.cache_data
def load_processed_data():
    df1, df2 = load_raw_data()
    df2["transmission"] = df2["id_transmission"].map({1: "manual", 2: "automatic"})
    df1["mileage (km)"] = df1["mileage (km)"] * 1000
    df = pd.concat([df1, df2], ignore_index=True)
    df = df[["brand", "model", "year", "mileage (km)", "transmission", "price (Rp)"]]
    df["brand"]        = df["brand"].astype(str).str.lower().str.strip()
    df["model"]        = df["model"].astype(str).str.lower().str.strip()
    df["transmission"] = df["transmission"].astype(str).str.lower().str.strip()
    return df

df1_raw, df2_raw = load_raw_data()
df               = load_processed_data()

# ── Model ──────────────────────────────────────────────────────────────────────
model         = joblib.load("model/model.pkl")
le_trans      = joblib.load("model/le_trans.pkl")
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

valid_brands    = df["brand"].unique()
valid_models    = df["model"].unique()
brand_model_map = df.groupby("brand")["model"].unique().to_dict()

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
        ["📂  Dataset", "📊  EDA", "⚙️  Preprocessing", "🧠  Training", "🔮  Prediction", "👥  About Us"],
        label_visibility="collapsed",
    )
    menu = menu.split("  ", 1)[1]

    st.markdown("<hr>", unsafe_allow_html=True)
    
    # GitHub button
    st.markdown("""
    <a href="https://github.com/jstn77/Used-Car-Price-Prediction" target="_blank" class="github-button">
        <span>🔗</span> GitHub Repository
    </a>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-footer">
        Group 02<br>
        Justin C.K. · Brian N.T. · Marvin A.R.<br>Jason B. · Kian A.W.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════
if menu == "Dataset":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-title">📂 Raw Dataset</div>
        <div class="page-header-purpose">Explore and understand the structure of the used car datasets before processing</div>
        <div class="page-header-steps">Step 1 of 6</div>
    </div>
    """, unsafe_allow_html=True)

    t1, t2 = st.tabs(["Dataset 1 — used_car.csv", "Dataset 2 — used_car_data_new.csv"])

    with t1:
        col_a, col_b, col_c = st.columns(3)
        col_a.markdown(f'<div class="metric-card"><div class="metric-label">Rows</div><div class="metric-value">{df1_raw.shape[0]:,}</div></div>', unsafe_allow_html=True)
        col_b.markdown(f'<div class="metric-card"><div class="metric-label">Columns</div><div class="metric-value">{df1_raw.shape[1]}</div></div>', unsafe_allow_html=True)
        col_c.markdown(f'<div class="metric-card"><div class="metric-label">Missing Values</div><div class="metric-value">{df1_raw.isnull().sum().sum()}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        f1, f2, f4 = st.columns([2, 4, 1])
        with f1:
            brands_ds1 = sorted(df1_raw["brand"].dropna().astype(str).str.lower().str.strip().unique().tolist()) if "brand" in df1_raw.columns else []
            brand_f1   = st.selectbox("Filter by brand", ["All"] + [b.title() for b in brands_ds1], key="ds1_brand")
        with f2:
            cols_sel1 = st.multiselect("Columns", df1_raw.columns.tolist(), default=df1_raw.columns.tolist(), key="ds1_cols")
        with f4:
            n_rows1 = st.number_input("Rows", min_value=10, max_value=5000, value=100, step=50, key="ds1_n")

        view1 = df1_raw[cols_sel1] if cols_sel1 else df1_raw
        if brand_f1 != "All" and "brand" in view1.columns:
            view1 = view1[view1["brand"].astype(str).str.lower().str.strip() == brand_f1.lower()]
        st.dataframe(view1.head(n_rows1), use_container_width=True)

    with t2:
        col_a, col_b, col_c = st.columns(3)
        col_a.markdown(f'<div class="metric-card"><div class="metric-label">Rows</div><div class="metric-value">{df2_raw.shape[0]:,}</div></div>', unsafe_allow_html=True)
        col_b.markdown(f'<div class="metric-card"><div class="metric-label">Columns</div><div class="metric-value">{df2_raw.shape[1]}</div></div>', unsafe_allow_html=True)
        col_c.markdown(f'<div class="metric-card"><div class="metric-label">Missing Values</div><div class="metric-value">{df2_raw.isnull().sum().sum()}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        f2, f4 = st.columns([4, 1])
        with f2:
            cols_sel2 = st.multiselect("Columns", df2_raw.columns.tolist(), default=df2_raw.columns.tolist(), key="ds2_cols")
        with f4:
            n_rows2 = st.number_input("Rows", min_value=10, max_value=5000, value=100, step=50, key="ds2_n")

        view2 = df2_raw[cols_sel2] if cols_sel2 else df2_raw
        st.dataframe(view2.head(n_rows2), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# EDA
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "EDA":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-title">📊 Exploratory Data Analysis</div>
        <div class="page-header-purpose">Analyze patterns, distributions, and relationships in the dataset</div>
        <div class="page-header-steps">Step 2 of 6</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Interactive filters ────────────────────────────────────────────────
    fa, fb, fc = st.columns([2, 2, 3])
    with fa:
        brand_options = ["All"] + sorted([b.title() for b in df["brand"].unique()])
        eda_brand = st.selectbox("Brand", brand_options, key="eda_brand")
    with fb:
        eda_trans = st.selectbox("Transmission", ["All", "Manual", "Automatic"], key="eda_trans")
    with fc:
        p_min, p_max = int(df["price (Rp)"].min()), int(df["price (Rp)"].max())
        eda_price = st.slider(
            "Price range (Rp)", p_min, p_max, (p_min, p_max),
            step=5_000_000, format="Rp %d", key="eda_price",
        )

    df_e = df[df["price (Rp)"].between(*eda_price)].copy()
    if eda_brand != "All":
        df_e = df_e[df_e["brand"] == eda_brand.lower()]
    if eda_trans != "All":
        df_e = df_e[df_e["transmission"] == eda_trans.lower()]

    # ── Overview metrics ───────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-label">Records</div><div class="metric-value">{df_e.shape[0]:,}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-label">Brands</div><div class="metric-value">{df_e["brand"].nunique()}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-label">Avg Price</div><div class="metric-value">Rp {df_e["price (Rp)"].mean()/1e6:.0f}M</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-label">Missing</div><div class="metric-value">{df_e.isnull().sum().sum()}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("Descriptive Statistics", expanded=False):
        st.dataframe(df_e.describe(), use_container_width=True)

    st.markdown("---")

    if df_e.empty:
        st.warning("No data matches the current filters.")
    else:
        # ── Row 1: Price dist + Top brands ────────────────────────────────
        col_l, col_r = st.columns(2)

        with col_l:
            show_price_dist = st.toggle("📊 Price Distribution", value=True, key="toggle_price_dist")
            if show_price_dist:
                st.markdown('<div class="chart-title">Price Distribution</div>', unsafe_allow_html=True)
                fig, ax = styled_fig(7, 4)
                sns.histplot(df_e["price (Rp)"], bins=40, kde=True, ax=ax,
                             color=ACCENT, alpha=0.55, line_kws={"linewidth": 1.5})
                mean_p   = df_e["price (Rp)"].mean()
                median_p = df_e["price (Rp)"].median()
                ax.axvline(mean_p,   color=AMBER,   linewidth=1.4, linestyle="--", alpha=0.9,
                           label=f"Mean Rp {mean_p/1e6:.0f}M")
                ax.axvline(median_p, color=EMERALD, linewidth=1.4, linestyle=":",  alpha=0.9,
                           label=f"Median Rp {median_p/1e6:.0f}M")
                ax.legend(fontsize=7.5, framealpha=0.15, labelcolor=TEXT_COLOR,
                          facecolor=CHART_BG, edgecolor=AXIS_COLOR)
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rp {x/1e6:.0f}M"))
                ax.set_xlabel("Price (Rp)")
                ax.set_ylabel("Count")
                ax.set_title("Car Price Distribution")
                apply_chart_style(fig, ax)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        with col_r:
            show_top_brands = st.toggle("📈 Top 10 Brands", value=True, key="toggle_top_brands")
            if show_top_brands:
                st.markdown('<div class="chart-title">Top 10 Brands by Listing</div>', unsafe_allow_html=True)
                fig, ax = styled_fig(7, 4)
                top_brands = df_e["brand"].value_counts().head(10)
                cols_b = bar_colors(len(top_brands))
                bars   = top_brands[::-1].plot(kind="barh", ax=ax, color=cols_b, zorder=2)
                add_bar_labels(ax, ax.patches, fmt="{:.0f}", color=TEXT_COLOR)
                ax.set_xlabel("Count")
                ax.set_title("Top 10 Brands by Listing")
                apply_chart_style(fig, ax)
                # clean right/top spines
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        # ── Row 2: Transmission donut + Year dist ─────────────────────────
        col_l, col_r = st.columns(2)

        with col_l:
            show_transmission = st.toggle("🔄 Transmission Split", value=True, key="toggle_transmission")
            if show_transmission:
                st.markdown('<div class="chart-title">Transmission Split</div>', unsafe_allow_html=True)
                fig, ax = styled_fig(7, 4)
                tc = df_e["transmission"].value_counts()
                wedge_colors = [ACCENT, ACCENT2]
                wedges, texts, autotexts = ax.pie(
                    tc, labels=[t.title() for t in tc.index],
                    autopct="%1.1f%%",
                    colors=wedge_colors[:len(tc)],
                    startangle=90,
                    textprops={"color": TEXT_COLOR, "fontsize": 9},
                    wedgeprops={"linewidth": 2.5, "edgecolor": CHART_BG, "width": 0.55},
                    pctdistance=0.78,
                )
                for at in autotexts:
                    at.set_fontsize(9)
                    at.set_color(FG_COLOR)
                ax.set_title("Manual vs Automatic")
                ax.title.set_color(FG_COLOR)
                ax.title.set_fontweight("600")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        with col_r:
            show_year = st.toggle("📅 Manufacturing Year", value=True, key="toggle_year")
            if show_year:
                st.markdown('<div class="chart-title">Manufacturing Year Distribution</div>', unsafe_allow_html=True)
                fig, ax = styled_fig(7, 4)
                sns.histplot(df_e["year"], bins=25, ax=ax, color=ACCENT2, alpha=0.65,
                             kde=True, line_kws={"linewidth": 1.4})
                med_year = int(df_e["year"].median())
                ax.axvline(med_year, color=AMBER, linewidth=1.4, linestyle="--",
                           label=f"Median {med_year}")
                ax.legend(fontsize=7.5, framealpha=0.15, labelcolor=TEXT_COLOR,
                          facecolor=CHART_BG, edgecolor=AXIS_COLOR)
                ax.set_xlabel("Year")
                ax.set_ylabel("Count")
                ax.set_title("Manufacturing Year Spread")
                apply_chart_style(fig, ax)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        # ── Row 3: Scatter plots ───────────────────────────────────────────
        col_l, col_r = st.columns(2)

        with col_l:
            show_mileage = st.toggle("🚗 Mileage vs Price", value=True, key="toggle_mileage")
            if show_mileage:
                st.markdown('<div class="chart-title">Mileage vs Price (coloured by Year)</div>', unsafe_allow_html=True)
                fig, ax = styled_fig(7, 4)
                sc = ax.scatter(
                    df_e["mileage (km)"], df_e["price (Rp)"],
                    c=df_e["year"], cmap="Blues" if LIGHT else "YlGnBu",
                    alpha=0.22, s=10, rasterized=True,
                    vmin=df_e["year"].min(), vmax=df_e["year"].max(),
                )
                cb = fig.colorbar(sc, ax=ax, fraction=0.028, pad=0.03)
                cb.set_label("Year", color=TEXT_COLOR, fontsize=8)
                cb.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelsize=7.5)
                cb.outline.set_edgecolor(AXIS_COLOR)
                plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rp {x/1e6:.0f}M"))
                ax.set_xlabel("Mileage (km)")
                ax.set_ylabel("Price (Rp)")
                ax.set_title("Mileage vs Price")
                apply_chart_style(fig, ax)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        with col_r:
            show_year_price = st.toggle("📊 Year vs Price", value=True, key="toggle_year_price")
            if show_year_price:
                st.markdown('<div class="chart-title">Year vs Price (coloured by Mileage)</div>', unsafe_allow_html=True)
                fig, ax = styled_fig(7, 4)
                sc = ax.scatter(
                    df_e["year"], df_e["price (Rp)"],
                    c=df_e["mileage (km)"], cmap="plasma_r",
                    alpha=0.22, s=10, rasterized=True,
                    vmin=df_e["mileage (km)"].quantile(0.05),
                    vmax=df_e["mileage (km)"].quantile(0.95),
                )
                cb = fig.colorbar(sc, ax=ax, fraction=0.028, pad=0.03)
                cb.set_label("Mileage (km)", color=TEXT_COLOR, fontsize=8)
                cb.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelsize=7.5)
                cb.outline.set_edgecolor(AXIS_COLOR)
                plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rp {x/1e6:.0f}M"))
                ax.set_xlabel("Year")
                ax.set_ylabel("Price (Rp)")
                ax.set_title("Year vs Price")
                apply_chart_style(fig, ax)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "Preprocessing":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-title">⚙️ Data Preprocessing</div>
        <div class="page-header-purpose">Clean data by removing missing values and duplicates for model training</div>
        <div class="page-header-steps">Step 3 of 6</div>
    </div>
    """, unsafe_allow_html=True)

    df_clean = df.dropna().drop_duplicates()

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-label">Before</div><div class="metric-value">{df.shape[0]:,}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-label">After</div><div class="metric-value">{df_clean.shape[0]:,}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-label">Removed</div><div class="metric-value">{df.shape[0] - df_clean.shape[0]:,}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-label">Retention</div><div class="metric-value">{df_clean.shape[0]/df.shape[0]*100:.1f}%</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Steps applied: <b>drop NaN rows</b> → <b>drop exact duplicates</b>
    → <b>lowercase &amp; trim</b> text columns (brand, model, transmission)
    </div>
    """, unsafe_allow_html=True)

    # Missing values per column chart
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    col_chart, col_table = st.columns([1, 2])

    with col_chart:
        if missing.empty:
            st.success("No missing values found in any column.")
        else:
            st.markdown('<div class="chart-title">Missing Values per Column</div>', unsafe_allow_html=True)
            fig, ax = styled_fig(4, max(2.5, len(missing) * 0.5))
            cols_m = bar_colors(len(missing), base="#f59e0b")
            missing.sort_values().plot(kind="barh", ax=ax, color=cols_m, zorder=2)
            add_bar_labels(ax, ax.patches, fmt="{:.0f}", color=TEXT_COLOR)
            ax.set_xlabel("Missing Count")
            ax.set_title("Missing by Column")
            apply_chart_style(fig, ax)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with col_table:
        st.markdown('<div class="chart-title">Cleaned Data Preview</div>', unsafe_allow_html=True)

        filter_brand = st.selectbox(
            "Filter by brand",
            ["All"] + sorted([b.title() for b in df_clean["brand"].unique()]),
            key="pp_brand",
        )
        view = df_clean if filter_brand == "All" else df_clean[df_clean["brand"] == filter_brand.lower()]
        st.dataframe(view.head(500), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "Training":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-title">🧠 Model Training & Evaluation</div>
        <div class="page-header-purpose">Build and evaluate a Random Forest model to predict used car prices</div>
        <div class="page-header-steps">Step 4 of 6</div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("#### Algorithm")
        st.markdown("""
        <div class="info-box">
        <b>Random Forest Regressor</b><br><br>
        An ensemble of decision trees that reduces overfitting by averaging
        predictions and handles non-linear patterns inherent in used-car pricing.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Training Pipeline")
        steps = [
            ("1", "Data Preparation",   "Merged two datasets, removed nulls & duplicates, normalised text"),
            ("2", "Feature Engineering","Derived <code>car_age = 2026 − year</code>"),
            ("3", "Encoding",           "Label Encoding → transmission · One-Hot Encoding → brand & model"),
            ("4", "Target Transform",   "Applied <code>log1p(price)</code> to reduce skewness"),
            ("5", "Data Split",         "80 % training · 20 % testing"),
            ("6", "Model Fit",          "RandomForestRegressor(n_estimators=150)"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div class="step-row">
                <div class="step-num">{num}</div>
                <div>
                    <div class="step-title">{title}</div>
                    <div class="step-desc">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with right:
        st.markdown("#### Model Performance")
        m1, m2, m3 = st.columns(3)
        m1.markdown('<div class="metric-card"><div class="metric-label">MAE</div><div class="metric-value">Rp 22.9M</div></div>', unsafe_allow_html=True)
        m2.markdown('<div class="metric-card"><div class="metric-label">RMSE</div><div class="metric-value">Rp 35.8M</div></div>', unsafe_allow_html=True)
        m3.markdown('<div class="metric-card"><div class="metric-label">R² Score</div><div class="metric-value">0.59</div></div>', unsafe_allow_html=True)
        st.caption("Evaluated on the test set after inverse log transformation.")

        st.markdown("<br>", unsafe_allow_html=True)
        top_n = st.slider("Top N features to display", min_value=5, max_value=20, value=10, key="fi_n")
        st.markdown(f'<div class="chart-title">Top {top_n} Feature Importances</div>', unsafe_allow_html=True)

        try:
            fi_df = pd.DataFrame({
                "feature":    model_columns,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False).head(top_n)

            fig, ax = styled_fig(6.5, max(3.5, top_n * 0.38))
            cols_fi = bar_colors(len(fi_df))
            ax.barh(fi_df["feature"], fi_df["importance"], color=cols_fi[::-1], zorder=2)
            ax.invert_yaxis()
            add_bar_labels(ax, ax.patches, fmt="{:.3f}", color=TEXT_COLOR)
            ax.set_xlabel("Importance Score")
            ax.set_title(f"Top {top_n} Feature Importances")
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            apply_chart_style(fig, ax)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception:
            st.warning("Feature importance not available.")

    st.markdown("---")
    st.success("**Conclusion:** Mileage and car age are the strongest predictors of price. Brand and model also carry significant weight.")

# ══════════════════════════════════════════════════════════════════════════════
# Prediction
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "Prediction":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-title">🔮 Price Prediction</div>
        <div class="page-header-purpose">Predict the market price of used cars based on vehicle specifications</div>
        <div class="page-header-steps">Step 5 of 6</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Enter your car details below. Brand and model names are matched with fuzzy logic — minor typos are handled automatically.
    </div>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            brand_input = st.text_input("Car Brand", placeholder="e.g. Toyota, Honda, Daihatsu")
            model_input = st.text_input("Car Model", placeholder="e.g. Innova, Brio, Agya")
        with c2:
            transmission = st.selectbox("Transmission", options=le_trans.classes_)
            year         = st.number_input("Manufacturing Year", min_value=2000, max_value=2026, value=2020)

        st.markdown("**Mileage (km)**")
        m1, m2 = st.columns([2, 1])
        with m1:
            mileage = st.slider("Slider", min_value=0, max_value=500_000, value=50_000, step=500,
                                format="%d km", label_visibility="collapsed")
        with m2:
            mileage = st.number_input("Manual Input", min_value=0, max_value=500_000, value=mileage, step=100,
                                      label_visibility="collapsed")

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

            car_age  = 2026 - year
            input_df = pd.DataFrame(0, index=[0], columns=model_columns)
            input_df["mileage (km)"] = mileage
            input_df["car_age"]      = car_age
            input_df["transmission"] = le_trans.transform([transmission])[0]

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
                candidates = [c.replace("model_", "") for c in model_columns if c.startswith("model_")]
                closest    = difflib.get_close_matches(model_match, candidates, n=1, cutoff=0.3)
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
                # Display prediction result first
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Estimated Market Price</div>
                    <div class="result-price">Rp {int(pred):,}</div>
                    <div class="result-sub">
                        {brand_match.title()} {model_match.title()} &nbsp;·&nbsp;
                        {year} &nbsp;·&nbsp; {mileage:,} km &nbsp;·&nbsp; {transmission.title()}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Then display car details
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.markdown(f'<div class="metric-card"><div class="metric-label">Brand</div><div class="metric-value" style="font-size:1.1rem">{brand_match.title()}</div></div>', unsafe_allow_html=True)
                col_b.markdown(f'<div class="metric-card"><div class="metric-label">Model</div><div class="metric-value" style="font-size:1.1rem">{model_match.title()}</div></div>', unsafe_allow_html=True)
                col_c.markdown(f'<div class="metric-card"><div class="metric-label">Year</div><div class="metric-value" style="font-size:1.1rem">{year}</div></div>', unsafe_allow_html=True)
                col_d.markdown(f'<div class="metric-card"><div class="metric-label">Mileage</div><div class="metric-value" style="font-size:1.1rem">{mileage:,} km</div></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Similar cars comparison chart
                similar = df[
                    (df["brand"] == brand_match) & (df["model"] == model_match)
                ]["price (Rp)"].dropna()

                if len(similar) >= 5:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="chart-title">How does this compare to similar listings?</div>', unsafe_allow_html=True)
                    fig, ax = styled_fig(9, 3.2)
                    n_bins = min(30, max(8, len(similar) // 5))
                    sns.histplot(similar, bins=n_bins, ax=ax, color=ACCENT, alpha=0.5,
                                 kde=True, line_kws={"linewidth": 1.5})
                    ax.axvline(pred, color=AMBER, linewidth=2, linestyle="--", zorder=5,
                               label=f"Your car: Rp {pred/1e6:.0f}M")
                    ax.axvline(similar.median(), color=EMERALD, linewidth=1.5, linestyle=":",
                               label=f"Market median: Rp {similar.median()/1e6:.0f}M")
                    ax.legend(fontsize=8, framealpha=0.15, labelcolor=TEXT_COLOR,
                              facecolor=CHART_BG, edgecolor=AXIS_COLOR)
                    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rp {x/1e6:.0f}M"))
                    ax.set_xlabel("Price (Rp)")
                    ax.set_ylabel("Count")
                    ax.set_title(f"Price distribution — {brand_match.title()} {model_match.title()} ({len(similar)} listings)")
                    ax.spines["right"].set_visible(False)
                    ax.spines["top"].set_visible(False)
                    apply_chart_style(fig, ax)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    pct = (similar < pred).mean() * 100
                    col_x, col_y, col_z = st.columns(3)
                    col_x.markdown(f'<div class="metric-card"><div class="metric-label">Listings Found</div><div class="metric-value">{len(similar)}</div></div>', unsafe_allow_html=True)
                    col_y.markdown(f'<div class="metric-card"><div class="metric-label">Market Median</div><div class="metric-value">Rp {similar.median()/1e6:.0f}M</div></div>', unsafe_allow_html=True)
                    col_z.markdown(f'<div class="metric-card"><div class="metric-label">Cheaper Than</div><div class="metric-value">{pct:.0f}%</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# About Us
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "About Us":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-title">👥 About Us</div>
        <div class="page-header-purpose">Meet the team behind this machine learning project and learn about our mission</div>
        <div class="page-header-steps">Step 6 of 6</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-banner">
        <div class="about-banner-title">Group 02 — Used Car Price Predictor</div>
        <div class="about-banner-sub">
            Students of Bina Nusantara University &nbsp;·&nbsp; Machine Learning Lecture
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_photo, col_desc = st.columns([1, 1], gap="large")

    with col_photo:
        st.image("us.jpeg", caption="Group 02 — Bina Nusantara University", use_container_width=True)

    with col_desc:
        st.markdown("""
        <div class="info-box">
            <b>About This Project</b><br><br>
            This web application was developed as part of the <b>Machine Learning</b> course at
            <b>Bina Nusantara University</b>. Our goal was to build an end-to-end ML pipeline
            that predicts the market price of used cars in Indonesia based on real-world listings.
        </div>
        <div class="info-box">
            <b>What We Built</b><br><br>
            We collected and merged two used-car datasets, performed exploratory data analysis,
            cleaned and engineered features, then trained a <b>Random Forest Regressor</b> to
            estimate prices. The entire workflow — from raw data to live predictions — is
            accessible through this interactive dashboard.
        </div>
        <div class="info-box">
            <b>Tech Stack</b><br><br>
            Python &nbsp;·&nbsp; Scikit-learn &nbsp;·&nbsp; Pandas &nbsp;·&nbsp;
            Matplotlib / Seaborn &nbsp;·&nbsp; Streamlit
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Meet the Team")

    members = [
        ("2802399463", "Justin Christian Kenan",   "🧑‍💻"),
        ("2802403183", "Brian Nicholas Tedjo",      "🧑‍💻"),
        ("2802402275", "Marvin Adriano Rusdianto",  "🧑‍💻"),
        ("2802419446", "Jason Budiharjo",           "🧑‍💻"),
        ("2802464582", "Kian Aurelio Wibowo",       "🧑‍💻"),
    ]

    cols = st.columns(5)
    for col, (sid, name, icon) in zip(cols, members):
        col.markdown(f"""
        <div class="member-card">
            <div class="member-avatar">{icon}</div>
            <div class="member-name">{name}</div>
            <div class="member-id">{sid}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;color:var(--subtle);font-size:0.8rem;">
        Bina Nusantara University &nbsp;·&nbsp; Machine Learning Lecture &nbsp;·&nbsp; 2025
    </div>
    """, unsafe_allow_html=True)
