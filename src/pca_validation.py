from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from sklearn.preprocessing import StandardScaler  
from utils.logger import Logger


def validar_supuestos_pca(
    df: pd.DataFrame,
    features: list[str],
    carpeta_reportes: str = "reports/figures",
) -> dict:
    """Evalúa viabilidad estadística de aplicar PCA."""
    Path(carpeta_reportes).mkdir(parents=True, exist_ok=True)
    X = df[features]
    scaler = StandardScaler()  # <-- CAMBIO
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
        Logger.info("Bartlett APROBADO (p < 0.05).")
    else:
        Logger.warning("Bartlett RECHAZADO (p >= 0.05).")

    kmo_por_var, kmo_global = calculate_kmo(X_scaled)
    Logger.info(f"Índice KMO Global: {kmo_global:.3f}")
    if kmo_global >= 0.60:
        Logger.info("KMO APROBADO (>= 0.60).")
    else:
        Logger.warning("KMO BAJO (< 0.60):")
        for var, val in zip(features, kmo_por_var):
            Logger.warning(f"  - {var}: {val:.3f}")

    vars_clave = [v for v in ["Retorno_Anual", "Volatilidad_Anual", "ROE",
                              "Deuda_Capital", "Beta"] if v in features]
    if len(vars_clave) >= 2:
        sns.pairplot(X_scaled[vars_clave], diag_kind="kde",
                     plot_kws={"alpha": 0.5, "s": 25})
        plt.savefig(f"{carpeta_reportes}/02_linealidad_pairplot.png", dpi=200)
        plt.close()

    return {
        "bartlett_chi2": float(chi2),
        "bartlett_p": float(p_value),
        "kmo_global": float(kmo_global),
        "kmo_por_variable": {v: float(k) for v, k in zip(features, kmo_por_var)},
    }