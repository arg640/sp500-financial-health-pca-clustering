from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from sklearn.preprocessing import RobustScaler
from utils.logger import Logger


def validar_supuestos_pca(
    df: pd.DataFrame,
    features: list[str],
    carpeta_reportes: str = "reports/figures",
) -> dict:
    """
    Evalúa la viabilidad estadística de aplicar PCA mediante correlaciones,
    Test de Esfericidad de Bartlett, índice KMO y gráficos de linealidad.

    Args:
        df (pd.DataFrame): Dataset financiero consolidado.
        features (list[str]): Lista de columnas numéricas a evaluar.
        carpeta_reportes (str): Directorio donde se guardarán las gráficas de diagnóstico. Defaults to "reports/figures".

    Returns:
        dict: Resultados numéricos de Bartlett (chi2, p_value) y KMO global.
    """
    Path(carpeta_reportes).mkdir(parents=True, exist_ok=True)
    X = df[features]
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features, index=X.index)
    corr = X_scaled.corr()
    plt.figure(figsize=(11, 9))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Matriz de Correlación - Salud Financiera y Riesgo")
    plt.tight_layout()
    plt.savefig(f"{carpeta_reportes}/01_matriz_correlacion.png", dpi=300)
    plt.close()
    chi2, p_value = calculate_bartlett_sphericity(X_scaled)
    Logger.info(f"Test de Bartlett -> Chi2: {chi2:.2f} | p-valor: {p_value:.4e}")
    if p_value < 0.05:
        Logger.info("Bartlett APROBADO (p < 0.05): Existen correlaciones significativas entre variables.")
    else:
        Logger.warning("Bartlett RECHAZADO (p >= 0.05): Las variables no están suficientemente correlacionadas.")
    kmo_por_var, kmo_global = calculate_kmo(X_scaled)
    Logger.info(f"Índice KMO Global: {kmo_global:.3f}")
    if kmo_global >= 0.60:
        Logger.info("KMO APROBADO (>= 0.60): Adecuación muestral válida para PCA.")
    else:
        Logger.warning("KMO BAJO (< 0.60): Revisa variables con KMO individual bajo:")
        for var, val in zip(features, kmo_por_var):
            Logger.warning(f"  - {var}: {val:.3f}")
    vars_clave = [v for v in ["Retorno_Anual", "Volatilidad_Anual", "ROE", "Deuda_Capital", "Beta"] if v in features]
    sns.pairplot(X_scaled[vars_clave], diag_kind="kde", plot_kws={"alpha": 0.5, "s": 25})
    plt.savefig(f"{carpeta_reportes}/02_linealidad_pairplot.png", dpi=200)
    plt.close()
    return {"bartlett_chi2": chi2, "bartlett_p": p_value, "kmo_global": kmo_global}