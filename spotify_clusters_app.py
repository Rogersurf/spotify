# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# --- Page Config ---
st.set_page_config(
    page_title="Spotify Song Clustering — Business Insights",
    page_icon="🎵",
    layout="wide",
)

# --- Title / Sidebar ---
st.title("🎵 Spotify Song Clustering — Business Insights Dashboard")
st.sidebar.header("Filters 📊")

# --- Data Loading & Caching (mantido) ---
@st.cache_data
def load_raw_data():
    url = 'https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv'
    return pd.read_csv(url)

df = load_raw_data()

# --- Business Problem Statement (mantido e no estilo do 1º app) ---
st.markdown("""
#### Business Problem
Spotify aims to deliver smarter recommendations and more engaging playlists by understanding the *mood* and *context* of songs, not just their genre.  
This dashboard uses **unsupervised learning** to reveal actionable clusters, helping Spotify personalize user experience, optimize curation, and unlock new business opportunities.
""")

with st.expander("📊 **Key Components of the Analysis**"):
    st.markdown("""
- **Audio features**: `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`.
- **Dimensionality reduction**: PCA for visualization (2D/3D) e melhor separação de grupos.
- **Clustering**: KMeans para identificar segmentos semelhantes por “mood/context”.
- **Business view**: Perfis de clusters com recomendações acionáveis.
""")

# --- Cleaning & Scaling (mantido) ---
@st.cache_data
def clean_and_scale(songs):
    drop_cols = [
        "track_id", "track_name", "track_artist", "track_album_id",
        "track_album_name", "track_album_release_date",
        "playlist_name", "playlist_id", "tempo", "duration_ms"
    ]
    songs_clean = songs.drop(columns=drop_cols, errors="ignore")
    features = [
        "danceability", "energy", "loudness", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence"
    ]
    X = songs_clean[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return songs, songs_clean, X, X_scaled, features

songs = load_raw_data()
songs, songs_clean, X, X_scaled, features = clean_and_scale(songs)

# --- Sidebar Filters (no estilo do 1º app; seguros quanto a colunas) ---
# Playlist genre filter
if "playlist_genre" in songs.columns:
    all_genres = sorted(list(pd.Series(songs["playlist_genre"].dropna().unique()).astype(str)))
else:
    all_genres = []
selected_genres = st.sidebar.multiselect("Select Playlist Genre 🎧", all_genres, default=all_genres if all_genres else None)

# Popularity range filter
if "track_popularity" in songs.columns:
    pop_min, pop_max = int(songs["track_popularity"].min()), int(songs["track_popularity"].max())
    pop_range = st.sidebar.slider("Track Popularity Range ⭐", pop_min, pop_max, (pop_min, pop_max))
else:
    pop_range = None

# Apply filters to an auxiliary DataFrame (apenas para visuais que usam 'songs')
songs_filtered = songs.copy()
if selected_genres:
    songs_filtered = songs_filtered[songs_filtered["playlist_genre"].astype(str).isin(selected_genres)]
if pop_range:
    songs_filtered = songs_filtered[(songs_filtered["track_popularity"] >= pop_range[0]) &
                                    (songs_filtered["track_popularity"] <= pop_range[1])]

# --- Controls for PCA & KMeans (mantidos) ---
st.sidebar.header("🔎 Explore Clusters")
n_components = st.sidebar.slider("PCA Components (for visualization)", 2, len(features), 3)
k_clusters = st.sidebar.slider("Number of clusters (KMeans)", 2, 15, 10)

@st.cache_data
def run_pca(X_scaled, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X_scaled)

@st.cache_data
def run_kmeans(X_pca, k_clusters):
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(X_pca)

X_pca = run_pca(X_scaled, n_components)
clusters = run_kmeans(X_pca, k_clusters)

songs_clustered = songs_clean.loc[X.index].copy()
songs_clustered["cluster"] = clusters

# --- Visualization Selector (como no 1º app) ---
st.header("Analysis 📊")
visualization_option = st.selectbox(
    "Select Visualization 🎨",
    [
        "2D PCA Scatter (by cluster)",
        "3D PCA Scatter (by cluster)",
        "Cluster Profiles — Average Audio Features (heatmap)",
        "Feature Distributions by Cluster (boxplots)",
        "Correlation Heatmap of Audio Features",
        "Are clusters separable by popularity? (Altair scatter)"
    ],
)

# --- Visualizations ---
if visualization_option == "2D PCA Scatter (by cluster)":
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette="tab10", s=12, ax=ax, legend=False)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("2D PCA — Songs clustered")
    st.pyplot(fig)

elif visualization_option == "3D PCA Scatter (by cluster)":
    if n_components >= 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=clusters, cmap='tab10', s=8)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title("3D PCA — Songs clustered")
        st.pyplot(fig)
    else:
        st.info("Select at least 3 PCA components for 3D visualization.")

elif visualization_option == "Cluster Profiles — Average Audio Features (heatmap)":
    st.subheader("Cluster Profiles — Average Audio Features")
    cluster_profile = songs_clustered.groupby("cluster")[features].mean().round(2)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(cluster_profile, annot=True, cmap="viridis", ax=ax)
    ax.set_title("Average Feature Values per Cluster")
    st.pyplot(fig)
    st.dataframe(cluster_profile)

elif visualization_option == "Feature Distributions by Cluster (boxplots)":
    st.subheader("Feature Distributions by Cluster")
    selected_feature = st.selectbox("Select feature", features, index=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=songs_clustered, x="cluster", y=selected_feature, ax=ax)
    ax.set_title(f"Distribution of {selected_feature} by cluster")
    st.pyplot(fig)

elif visualization_option == "Correlation Heatmap of Audio Features":
    st.subheader("Correlation Heatmap")
    corr = pd.DataFrame(X, columns=features).corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation among audio features")
    st.pyplot(fig)

elif visualization_option == "Are clusters separable by popularity? (Altair scatter)":
    # Só usa se houver track_popularity; mapeia com as amostras do cluster
    if "track_popularity" in songs.columns:
        # alinhar índices: precisamos trazer popularity de 'songs' para 'songs_clustered'
        pop_series = songs.loc[songs_clustered.index, "track_popularity"] if "track_popularity" in songs.columns else pd.Series(index=songs_clustered.index, dtype=float)
        viz_df = songs_clustered.copy()
        viz_df["track_popularity"] = pop_series
        viz_df = viz_df.dropna(subset=["track_popularity"])
        chart = alt.Chart(viz_df.reset_index(drop=True)).mark_point(filled=True).encode(
            alt.X('track_popularity:Q', title='Track popularity'),
            alt.Y('valence:Q', title='Valence'),
            alt.Color('cluster:N'),
            alt.OpacityValue(0.7),
            tooltip=['cluster:N'] + [c for c in ["valence", "energy", "danceability"] if c in viz_df.columns]
        ).properties(height=450)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("`track_popularity` not available in this dataset.")

# --- Cluster Insights (mantidos) ---
st.sidebar.markdown("---")
selected_cluster = st.sidebar.selectbox("Select cluster for details", sorted(songs_clustered["cluster"].unique()))

cluster_business = {
    0: "Acoustic / Chill 🌿: Calm, relaxing. Use in study/wellness playlists.",
    1: "Classic Rock 🎸: Guitar-driven, nostalgic. Promote with live events.",
    2: "EDM / Dance 🎧: High-energy. Add to workout/party playlists.",
    3: "Electropop / Dance Pop 🔥: Catchy, upbeat. Viral playlists, social media.",
    4: "Hard Rock / Metal 🤘: Loud, intense. Niche playlists, festival tie-ins.",
    5: "Indie / Alternative 🌌: Creative, experimental. Discovery playlists.",
    6: "Latin / Reggaeton 🌴: Rhythmic, upbeat. Geo-targeted playlists.",
    7: "Pop Mainstream 🎶: Balanced, mass-market. Chart-topping hits.",
    8: "R&B / Soul 💜: Smooth, emotional. Romance/mood playlists.",
    9: "Rap / Trap 🎤: Speech-heavy, beat-driven. Youth/urban playlists."
}

cluster_actions = {
    0: "Focus/study playlists, wellness app partnerships.",
    1: "Live concert tie-ins, nostalgic campaigns.",
    2: "Workout/party playlists, fitness brand collaborations.",
    3: "Viral playlist promotion, social media campaigns.",
    4: "Niche playlist curation, festival partnerships.",
    5: "Discovery playlists, support for emerging artists.",
    6: "Geo-targeted playlists, dance event promotions.",
    7: "Algorithmic playlist anchors, sponsored content.",
    8: "Romance/mood playlists, lifestyle brand partnerships.",
    9: "Youth/urban playlists, influencer collaborations."
}

st.markdown(f"### Cluster {selected_cluster} — Business Insights")
st.markdown(f"**Business Description:** {cluster_business.get(selected_cluster, 'Segmented by mood/context.')}")
st.markdown(f"**Actionable Recommendation:** {cluster_actions.get(selected_cluster, 'Curate and promote according to cluster characteristics.')}")
st.markdown("**Sample Songs in this Cluster:**")
sample_cols = [c for c in ["track_name", "track_artist", "playlist_genre"] if c in songs.columns]
st.write(songs.loc[songs_clustered[songs_clustered['cluster'] == selected_cluster].index, sample_cols].head(10))

# --- Dataset Overview (no estilo do 1º app) ---
st.header("Dataset Overview")
st.dataframe(songs.describe(include='all').transpose())

# --- Insights Expander (no estilo do 1º app) ---
with st.expander("Interpreting the visualizations"):
    st.markdown("""
1. **PCA scatter** — clusters tendem a ocupar regiões distintas, sugerindo *moods* diferentes (e.g., alto `energy` + baixo `acousticness` próximos).
2. **Heatmap de perfis** — médias por cluster evidenciam contrastes claros (e.g., `valence`/`danceability` altos para pop/dance).
3. **Boxplots por cluster** — mostram dispersão e outliers por feature, úteis para ajustar K ou features.
4. **Correlação** — `energy` e `loudness` costumam correlacionar; atenção ao leakage de escala.
5. **Popularidade vs. valence** — clusters com maior `valence`/`danceability` podem apresentar popularidade maior (insight para marketing).
""")

# --- Rationale & Strategic Insights (mantido) ---
st.header("Rationale & Strategic Insights")
st.markdown("""
### Why This Approach?
- **Business Need:** Genre-based recommendations miss nuances of mood/context. Clustering por audio features revela segmentos acionáveis.
- **Data Cleaning:** Remoção de colunas não-audio e linhas incompletas.
- **Feature Selection:** Foco nos 8 atributos mais ligados a “mood/context”.
- **Scaling:** `StandardScaler` equilibra contribuições.
- **PCA:** Visualização e separação mais clara.
- **KMeans:** Grupos interpretáveis para ação.

### Strategic Insights for Stakeholders
- **Personalization:** Recomendações contextuais aumentam relevância e satisfação.
- **Playlist Curation:** Atribuição automática de novas faixas a clusters específicos.
- **Marketing & Engagement:** Campanhas e playlists temáticas por cluster.
- **Artist Discovery:** Tendências emergentes por segmento.
- **Partnerships & Revenue:** Parcerias alinhadas ao *mood* (wellness, fitness, etc.).
- **Continuous Improvement:** Medir performance por cluster → iterar K, features e regras.
""")

# --- Footer ---
st.markdown("---")
st.markdown("© 2025 Spotify Clustering Assignment — Powered by Streamlit")
