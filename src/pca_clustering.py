from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler
from utils.logger import Logger


def evaluar_numero_clusters(componentes: np.ndarray, k_max: int = 7, carpeta_reportes: str = "reports/figures") -> None:
    """
    Genera la gráfica del Método del Codo (Inercia) y Silhouette Score para elegir el K óptimo.

    Args:
        componentes (np.ndarray): Coordenadas proyectadas en el espacio PCA.
        k_max (int): Número máximo de clusters a evaluar. Defaults to 7.
        carpeta_reportes (str): Directorio de salida para la gráfica. Defaults to "reports/figures".
    """
    Path(carpeta_reportes).mkdir(parents=True, exist_ok=True)
    ks = list(range(2, k_max + 1))
    inercias, silhouettes = [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        etiquetas = km.fit_predict(componentes)
        inercias.append(km.inertia_)
        silhouettes.append(silhouette_score(componentes, etiquetas))
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
    n_clusters: int = 4,
    carpeta_reportes: str = "reports/figures",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta PCA (2 componentes) y K-Means para construir el mapa de riesgo-rentabilidad.

    Args:
        df (pd.DataFrame): Dataset financiero consolidado.
        features (list[str]): Variables numéricas incluidas en el modelo.
        n_clusters (int): Cantidad de grupos para clasificar las empresas. Defaults to 4.
        carpeta_reportes (str): Directorio de salida para visualizaciones. Defaults to "reports/figures".

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: DataFrame original enriquecido con PC1, PC2 y Cluster,
        y DataFrame con las cargas (loadings) de cada variable.
    """
    Path(carpeta_reportes).mkdir(parents=True, exist_ok=True)
    X_scaled = RobustScaler().fit_transform(df[features])
    pca = PCA(n_components=2, random_state=42)
    componentes = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_
    Logger.info(f"Varianza explicada -> PC1: {var_exp[0]:.2%} | PC2: {var_exp[1]:.2%} | Total: {sum(var_exp):.2%}")
    loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=features)
    evaluar_numero_clusters(componentes, k_max=7, carpeta_reportes=carpeta_reportes)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_resultado = df.copy()
    df_resultado["PC1"] = componentes[:, 0]
    df_resultado["PC2"] = componentes[:, 1]
    df_resultado["Cluster"] = kmeans.fit_predict(componentes)
    plt.figure(figsize=(14, 9))
    sns.scatterplot(
        data=df_resultado,
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette="viridis",
        s=90,
        alpha=0.85,
    )
    escala = np.max(np.abs(componentes)) * 0.75
    for i, var in enumerate(features):
        dx, dy = loadings.iloc[i, 0] * escala, loadings.iloc[i, 1] * escala
        plt.arrow(0, 0, dx, dy, color="crimson", alpha=0.65, head_width=0.15)
        plt.text(dx * 1.12, dy * 1.12, var, color="darkred", fontsize=9, fontweight="bold")
    lim_x = np.percentile(np.abs(df_resultado["PC1"]), 92)
    lim_y = np.percentile(np.abs(df_resultado["PC2"]), 92)
    for ticker, fila in df_resultado.iterrows():
        if abs(fila["PC1"]) > lim_x or abs(fila["PC2"]) > lim_y:
            plt.text(fila["PC1"] + 0.08, fila["PC2"] + 0.08, ticker, fontsize=8, alpha=0.85)
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.title(f"Mapa Riesgo-Rentabilidad y Salud Financiera (PCA + K-Means)\nVarianza Explicada: {sum(var_exp):.1%}")
    plt.xlabel(f"Componente Principal 1 ({var_exp[0]:.1%} varianza)")
    plt.ylabel(f"Componente Principal 2 ({var_exp[1]:.1%} varianza)")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig(f"{carpeta_reportes}/04_mapa_riesgo_rentabilidad.png", dpi=300)
    plt.close()
    Logger.info(f"Mapa Riesgo-Rentabilidad guardado en '{carpeta_reportes}/04_mapa_riesgo_rentabilidad.png'")
    return df_resultado, loadings