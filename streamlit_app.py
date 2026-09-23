"""
HitSense & PublicPay — interactive dashboards
Run locally with:  streamlit run streamlit_app.py
Deploy free at:     https://streamlit.io/cloud (connect this file + requirements.txt + the 3 CSVs to a GitHub repo)

Two tabs = two "single-interface" dashboards, one per dataset, each with a dynamic selector
that re-renders the visualizations live (the assignment's "layer in the interaction" requirement).
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="HitSense & PublicPay", layout="wide")

# ----------------------------------------------------------------------------
# Data loading (cached so re-running the app doesn't re-read the CSVs every click)
# ----------------------------------------------------------------------------
@st.cache_data
def load_spotify():
    tracks = pd.read_csv("dataset.csv")
    tracks = tracks.drop(columns=[c for c in tracks.columns if c.startswith("Unnamed")])
    return tracks


@st.cache_data
def load_worldbank():
    wwbi = pd.read_csv("WWBICSV.csv")
    country = pd.read_csv("WWBICountry.csv")
    year_columns = [str(y) for y in range(2000, 2023)]

    def latest_value(indicator_name):
        rows = wwbi[wwbi["Indicator Name"] == indicator_name].copy()

        def pick_latest(row):
            for year in reversed(year_columns):
                if pd.notna(row[year]):
                    return pd.Series([row[year], int(year)])
            return pd.Series([np.nan, np.nan])

        rows[["value", "year"]] = rows[year_columns].apply(pick_latest, axis=1)
        return rows[["Country Code", "Country Name", "value", "year"]].dropna(subset=["value"])

    wage = latest_value("Wage bill as a percentage of GDP").rename(columns={"value": "wage_pct_gdp"})
    wage = wage.merge(country[["Country Code", "Region", "Income Group"]], on="Country Code", how="left")
    wage = wage.dropna(subset=["Income Group"])

    employment = latest_value("Public sector employment, as a share of paid employment").rename(
        columns={"value": "pubsector_emp_share"}
    )
    combined = wage.merge(employment[["Country Code", "pubsector_emp_share"]], on="Country Code", how="left")
    return combined


INCOME_ORDER = ["Low income", "Lower middle income", "Upper middle income", "High income"]
PALETTE = {"Low income": "#f4a261", "Lower middle income": "#e76f51",
           "Upper middle income": "#2a9d8f", "High income": "#264653"}

tab1, tab2 = st.tabs(["🎵 HitSense — Spotify", "🏛️ PublicPay — World Bank"])

# ============================================================================
# TAB 1 — SPOTIFY
# ============================================================================
with tab1:
    st.header("What actually makes a song popular?")
    st.markdown(
        "Audio features (danceability, energy, loudness...) correlate weakly with popularity "
        "(|r| < 0.10 for every feature). Genre explains far more. Use the controls to explore both."
    )

    tracks = load_spotify()

    col_a, col_b = st.columns([1, 1])
    with col_a:
        min_tracks = st.slider("Minimum tracks per genre (for stable averages)", 100, 1000, 500, step=100)
    with col_b:
        hit_threshold = st.slider("Popularity threshold that defines a 'hit'", 40, 90, 70, step=5)

    genre_stats = tracks.groupby("track_genre")["popularity"].agg(mean_popularity="mean", track_count="size")
    genre_stats = genre_stats[genre_stats["track_count"] >= min_tracks]
    all_genres = sorted(genre_stats.index.tolist())
    selected_genres = st.multiselect(
        "Highlight specific genres in the bar chart (leave empty to show top/bottom 10)",
        options=all_genres,
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Genre vs. mean popularity")
        if selected_genres:
            chart_data = genre_stats.loc[selected_genres].sort_values("mean_popularity")
            colors = ["#468189"] * len(chart_data)
        else:
            top10 = genre_stats.sort_values("mean_popularity", ascending=False).head(10)
            bottom10 = genre_stats.sort_values("mean_popularity", ascending=False).tail(10)
            chart_data = pd.concat([top10, bottom10]).sort_values("mean_popularity")
            colors = ["#d1495b" if g in top10.index else "#468189" for g in chart_data.index]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.barh(chart_data.index, chart_data["mean_popularity"], color=colors)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Mean popularity (0-100)")
        st.pyplot(fig)

    with right:
        st.subheader(f"Audio features: hit (≥{hit_threshold}) vs. non-hit")
        tracks["tier"] = np.where(tracks["popularity"] >= hit_threshold, "Hit", "Non-hit")
        st.caption(f"{(tracks['tier'] == 'Hit').sum():,} hit tracks vs. {(tracks['tier'] == 'Non-hit').sum():,} non-hit tracks")

        feature = st.selectbox("Feature to compare", ["danceability", "energy", "loudness", "valence", "tempo"])
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        groups = [tracks.loc[tracks["tier"] == t, feature].dropna() for t in ["Non-hit", "Hit"]]
        box = ax2.boxplot(groups, tick_labels=["Non-hit", "Hit"], patch_artist=True, showfliers=False)
        for patch, color in zip(box["boxes"], ["#468189", "#d1495b"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax2.set_title(feature.capitalize())
        st.pyplot(fig2)

    st.subheader("Loudness vs. popularity (sampled for readability)")
    sample = tracks.sample(n=min(6000, len(tracks)), random_state=7)
    corr = tracks["loudness"].corr(tracks["popularity"])
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.scatter(sample["loudness"], sample["popularity"], s=8, alpha=0.15, color="#2e4057")
    ax3.set_xlabel("Loudness (dB)")
    ax3.set_ylabel("Popularity")
    ax3.set_title(f"Pearson r = {corr:.3f}")
    st.pyplot(fig3)

# ============================================================================
# TAB 2 — WORLD BANK
# ============================================================================
with tab2:
    st.header("Does national income predict public-sector pay?")
    st.markdown(
        "Wage bill as a share of GDP rises with income group, but workforce size alone is a weak predictor. "
        "Filter by income group or region below."
    )

    wb = load_worldbank()

    selected_income = st.multiselect("Income groups to include", INCOME_ORDER, default=INCOME_ORDER)
    selected_regions = st.multiselect(
        "Regions to include (optional filter)",
        sorted(wb["Region"].dropna().unique().tolist()),
        default=[],
    )

    filtered = wb[wb["Income Group"].isin(selected_income)]
    if selected_regions:
        filtered = filtered[filtered["Region"].isin(selected_regions)]

    st.caption(f"{len(filtered)} countries match the current filters")

    left2, right2 = st.columns(2)

    with left2:
        st.subheader("Median wage bill (% of GDP) by income group")
        medians = filtered.groupby("Income Group", observed=True)["wage_pct_gdp"].median()
        medians = medians.reindex([g for g in INCOME_ORDER if g in medians.index])
        fig4, ax4 = plt.subplots(figsize=(6, 5))
        ax4.bar(medians.index.astype(str), medians.values,
                color=[PALETTE[g] for g in medians.index])
        ax4.set_ylabel("Median wage bill (% of GDP)")
        plt.setp(ax4.get_xticklabels(), rotation=15)
        st.pyplot(fig4)

    with right2:
        st.subheader("Distribution by income group")
        fig5, ax5 = plt.subplots(figsize=(6, 5))
        present_groups = [g for g in INCOME_ORDER if g in filtered["Income Group"].unique()]
        grouped_vals = [filtered.loc[filtered["Income Group"] == g, "wage_pct_gdp"].dropna() for g in present_groups]
        box2 = ax5.boxplot(grouped_vals, tick_labels=present_groups, patch_artist=True)
        for patch, g in zip(box2["boxes"], present_groups):
            patch.set_facecolor(PALETTE[g])
            patch.set_alpha(0.8)
        ax5.set_ylabel("Wage bill (% of GDP)")
        plt.setp(ax5.get_xticklabels(), rotation=15)
        st.pyplot(fig5)

    st.subheader("Public-sector employment share vs. wage bill")
    scatter_data = filtered.dropna(subset=["pubsector_emp_share"])
    corr2 = scatter_data["pubsector_emp_share"].corr(scatter_data["wage_pct_gdp"]) if len(scatter_data) > 2 else float("nan")
    fig6, ax6 = plt.subplots(figsize=(10, 4.5))
    for g in INCOME_ORDER:
        sub = scatter_data[scatter_data["Income Group"] == g]
        if len(sub):
            ax6.scatter(sub["pubsector_emp_share"] * 100, sub["wage_pct_gdp"],
                        label=g, color=PALETTE[g], alpha=0.85, s=45, edgecolor="white", linewidth=0.4)
    ax6.set_xlabel("Public-sector employment (% of paid employment)")
    ax6.set_ylabel("Wage bill (% of GDP)")
    ax6.set_title(f"Pearson r = {corr2:.2f} across {len(scatter_data)} countries" if scatter_data.notna().any().any() else "Not enough data for current filters")
    ax6.legend(title="Income group", fontsize=8)
    st.pyplot(fig6)
