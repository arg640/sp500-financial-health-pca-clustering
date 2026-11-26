from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler  
from utils.logger import Logger


def heatmap_silhouette_pc_k(
    X_scaled: np.ndarray,
    pc_rango=range(2, 12),
    k_rango=range(2, 8),
    carpeta_reportes: str = "reports/figures",
) -> pd.DataFrame:
    """Fig. 5 del paper: silhouette para cada combinación (n_PCs, K)."""
    Path(carpeta_reportes).mkdir(parents=True, exist_ok=True)
    n_max = min(max(pc_rango), X_scaled.shape[1], X_scaled.shape[0] - 1)
    pc_rango = [p for p in pc_rango if p <= n_max]
    scores = np.full((len(pc_rango), len(k_rango)), np.nan)
    for i, n_pc in enumerate(pc_rango):
        Z = PCA(n_components=n_pc, random_state=42).fit_transform(X_scaled)
        for j, k in enumerate(k_rango):
            etq = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Z)
            scores[i, j] = silhouette_score(Z, etq)
    df_scores = pd.DataFrame(scores, index=pc_rango, columns=list(k_rango))
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_scores, annot=True, fmt=".3f", cmap="YlGnBu",
                cbar_kws={"label": "Silhouette Score"})
    plt.title("Silhouette Score by No. of PCA Components and Clusters (K)")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Number of Principal Components")
    plt.tight_layout()
    plt.savefig(f"{carpeta_reportes}/05_heatmap_silhouette_pc_k.png", dpi=300)
    plt.close()
    Logger.info(f"Heatmap silhouette guardado en '{carpeta_reportes}/05_heatmap_silhouette_pc_k.png'")
    for n_pc in pc_rango:
        fila = df_scores.loc[n_pc]
        Logger.info(f"  {n_pc} PCs -> mejor K = {fila.idxmax()} (silhouette = {fila.max():.3f})")
    return df_scores


def evaluar_numero_clusters(componentes: np.ndarray, k_max: int = 7,
                            carpeta_reportes: str = "reports/figures") -> None:
    """Genera codo + silhouette para elegir K."""
    Path(carpeta_reportes).mkdir(parents=True, exist_ok=True)
    ks = list(range(2, k_max + 1))
    inercias, silhouettes = [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        etq = km.fit_predict(componentes)
        inercias.append(km.inertia_)
        silhouettes.append(silhouette_score(componentes, etq))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(ks, inercias, marker="o", color="steelblue")
    ax1.set_title("Método del Codo (Inercia)")
    ax1.set_xlabel("Número de Clusters (K)")
    ax1.set_ylabel("Inercia")
    ax2.plot(ks, silhouettes, marker="s", color="darkorange")
    ax2.set_title("Índice de Silueta (Silhouette Score)")
    ax2.set_xlabel("Número de Clusters (K)")
    ax2.set_ylabel("Silhouette Score")
    plt.tight_layout()
    plt.savefig(f"{carpeta_reportes}/03_seleccion_k_clusters.png", dpi=300)
    plt.close()
    Logger.info(f"Gráfica de selección de clusters guardada en '{carpeta_reportes}/03_seleccion_k_clusters.png'")


def ejecutar_pca_clustering(
    df: pd.DataFrame,
    features: list[str],
    n_components: int = 6,
    n_clusters: int = 2,
    carpeta_reportes: str = "reports/figures",
    carpeta_tablas: str = "data/processed",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PCA(n_components) + K-Means sobre el espacio PCA."""
    Path(carpeta_reportes).mkdir(parents=True, exist_ok=True)
    Path(carpeta_tablas).mkdir(parents=True, exist_ok=True)

    # StandardScaler alineado con la ecuación de estandarización del paper
    X_scaled = StandardScaler().fit_transform(df[features])

    # Heatmap silhouette (Fig. 5)
    df_sil = heatmap_silhouette_pc_k(X_scaled, carpeta_reportes=carpeta_reportes)
    df_sil.to_csv(f"{carpeta_tablas}/silhouette_pc_k.csv")

    # PCA
    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_
    Logger.info(f"Varianza explicada con {n_components} PCs: {var_exp.sum():.2%}")
    Logger.info(f"PC1: {var_exp[0]:.2%} | PC2: {var_exp[1]:.2%}")

    # Loadings como correlaciones
    loadings = pd.DataFrame(
        pca.components_.T * np.sqrt(pca.explained_variance_),
        columns=[f"PC{i + 1}" for i in range(n_components)],
        index=features,
    )
    loadings.to_csv(f"{carpeta_tablas}/loadings_pca.csv")
    Logger.info(f"Loadings guardados en '{carpeta_tablas}/loadings_pca.csv'")

    # Codo + silhouette (K variable)
    evaluar_numero_clusters(Z, k_max=7, carpeta_reportes=carpeta_reportes)

    # K-Means final
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_resultado = df.copy()
    df_resultado["Cluster"] = kmeans.fit_predict(Z)
    sil = silhouette_score(Z, df_resultado["Cluster"])
    Logger.info(f"Silhouette final (n_components={n_components}, K={n_clusters}): {sil:.3f}")

    tamanos = df_resultado["Cluster"].value_counts().sort_index().to_dict()
    Logger.info(f"Tamaños de cluster: {tamanos}")

    # Figura: scatter volatilidad vs retorno
    plt.figure(figsize=(13, 8))
    sns.scatterplot(
        data=df_resultado, x="Volatilidad_Anual", y="Retorno_Anual",
        hue="Cluster", palette="viridis", s=90, alpha=0.85,
    )
    plt.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    plt.title(f"Mapa Riesgo–Retorno (PCA {n_components}D + K-Means, K={n_clusters}, silhouette={sil:.3f})")
    plt.xlabel("Volatilidad Anual (riesgo)")
    plt.ylabel("Retorno Anual")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig(f"{carpeta_reportes}/04_mapa_riesgo_retorno.png", dpi=300)
    plt.close()
    Logger.info(f"Mapa riesgo-retorno guardado en '{carpeta_reportes}/04_mapa_riesgo_retorno.png'")

    # Figura: heatmap Z-scores por cluster (Fig. principal)
    z = df_resultado.copy()
    z[features] = (z[features] - z[features].mean()) / z[features].std()
    perfil_z = z.groupby("Cluster")[features].mean().T
    plt.figure(figsize=(max(6, 2 + n_clusters * 1.5), 8))
    sns.heatmap(perfil_z, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-2, vmax=2, cbar_kws={"label": "Z-score"})
    plt.title("Perfil financiero por cluster (Z-scores sobre las 11 variables)")
    plt.xlabel("Cluster")
    plt.ylabel("Variable financiera")
    plt.tight_layout()
    plt.savefig(f"{carpeta_reportes}/07_heatmap_zscore_clusters.png", dpi=300)
    plt.close()
    Logger.info(f"Heatmap Z-scores guardado en '{carpeta_reportes}/07_heatmap_zscore_clusters.png'")

    # Figura: composición sectorial
    if "Sector" in df_resultado.columns:
        comp = pd.crosstab(df_resultado["Cluster"], df_resultado["Sector"])
        pct = comp.div(comp.sum(axis=1), axis=0) * 100
        plt.figure(figsize=(12, 6))
        pct.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="tab20")
        plt.title("Composición sectorial por cluster (%)")
        plt.xlabel("Cluster")
        plt.ylabel("Porcentaje (%)")
        plt.legend(title="Sector", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{carpeta_reportes}/08_composicion_sectorial.png", dpi=300)
        plt.close()
        Logger.info(f"Composición sectorial guardada en '{carpeta_reportes}/08_composicion_sectorial.png'")

    return df_resultado, loadings