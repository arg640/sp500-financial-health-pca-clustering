"""
Pipeline Integrado de Análisis de Componentes Principales (PCA), Clustering y Machine Learning
============================================================================================
Proyecto: Evaluación de Salud Financiera y Riesgo de Empresas del S&P 500
Objetivo: Consolidar en un solo script de Python todos los módulos del proyecto
          (scraping, limpieza, ingeniería de variables, validación PCA, clustering K-Means)
          y añadir las visualizaciones de PCA (varianza explicada, scree plot,
          mapa de calor de cargas, riesgo vs. retorno) junto con un módulo completo de
          Machine Learning (entrenamiento con 6 componentes principales, evaluación,
          comparativa PCA vs. variables originales, visualización e iteración).

Nota sobre ejecución en entornos sin conexión o con datos locales:
- Si existen archivos CSV previos en `data/raw/` o `data/processed/`, el script puede reutilizarlos.
- Si se requiere correr sin conexión externa o probar rápidamente, se incluye un modo opcional
  de respaldo sintético/local solo si falla la descarga de red y no hay CSV previo.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

try:
    from rich.logging import RichHandler

    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

try:
    from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

    _HAS_FACTOR_ANALYZER = True
except ImportError:
    _HAS_FACTOR_ANALYZER = False

try:
    import yfinance as yf

    _HAS_YFINANCE = True
except ImportError:
    _HAS_YFINANCE = False


FEATURES_FINANCIERAS: list[str] = [
    "Retorno_Anual",
    "Volatilidad_Anual",
    "Sharpe_Ratio",
    "Max_Drawdown",
    "Deuda_Capital",
    "Liquidez_Corriente",
    "ROE",
    "ROA",
    "Margen_Neto",
    "Beta",
    "PER",
]


# ==============================================================================
# SECCIÓN 1: LOGGING Y UTILIDADES DE E/S
# ==============================================================================
log = logging.getLogger("pca_sp500_pipeline")
log.setLevel(logging.INFO)
log.propagate = False
log.handlers.clear()

if _HAS_RICH:
    _handler: logging.Handler = RichHandler(
        markup=True,
        rich_tracebacks=True,
        show_time=True,
        show_level=False,
        show_path=False,
    )
    _handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
else:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    )

log.addHandler(_handler)


class Logger:
    """Wrapper de registro unificado para el pipeline."""

    @staticmethod
    def info(message: Any) -> None:
        log.info(f"{message}")

    @staticmethod
    def warning(message: Any) -> None:
        if _HAS_RICH:
            log.warning(f"[bold yellow]{message}[/]")
        else:
            log.warning(message)

    @staticmethod
    def error(message: Any) -> None:
        if _HAS_RICH:
            log.error(f"[bold red]{message}[/]")
        else:
            log.error(message)

    @staticmethod
    def critical(message: Any) -> None:
        if _HAS_RICH:
            log.critical(f"[bold red on white]{message}[/]")
        else:
            log.critical(message)


def guardar_csv(df: pd.DataFrame, ruta: str | Path, index: bool = True) -> None:
    """Crea el directorio si no existe y guarda el DataFrame en formato CSV."""
    ruta_path = Path(ruta)
    ruta_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta_path, index=index)
    Logger.info(f"Archivo guardado en '{ruta_path}' ({df.shape[0]} filas x {df.shape[1]} columnas)")


# ==============================================================================
# SECCIÓN 2: RECOLECCIÓN DE DATOS / SCRAPING
# ==============================================================================
def obtener_tickers_sp500() -> list[str]:
    """Obtiene y normaliza la lista vigente de tickers del S&P 500 desde Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tablas = pd.read_html(url, storage_options={"User-Agent": "Mozilla/5.0"})
    if not tablas:
        raise ValueError("No se encontraron tablas en la página de Wikipedia del S&P 500.")
    df = tablas[0]
    tickers = (
        df["Symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(".", "-", regex=False)
    )
    tickers_limpios = sorted(t for t in tickers.unique().tolist() if t and t != "NAN")
    Logger.info(f"Se obtuvieron {len(tickers_limpios)} tickers del S&P 500.")
    return tickers_limpios


def descargar_precios(empresas: list[str], periodo: str = "5y") -> pd.DataFrame:
    """Descarga precios de cierre ajustados y elimina columnas vacías o duplicadas."""
    if not _HAS_YFINANCE:
        raise RuntimeError("La librería `yfinance` no está instalada en el entorno.")
    if not empresas:
        raise ValueError("La lista de empresas está vacía.")

    Logger.info(f"Descargando histórico ({periodo}) de {len(empresas)} empresas...")
    data = yf.download(
        empresas,
        period=periodo,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    if data is None or data.empty:
        raise RuntimeError("No se recibieron datos de precios desde yfinance.")

    if isinstance(data.columns, pd.MultiIndex):
        nivel_1 = data.columns.get_level_values(1)
        if "Close" in nivel_1:
            precios = data.xs("Close", axis=1, level=1)
        elif "Adj Close" in nivel_1:
            precios = data.xs("Adj Close", axis=1, level=1)
        else:
            precios = data.xs(nivel_1[0], axis=1, level=1)
    else:
        if len(empresas) == 1 and "Close" in data.columns:
            precios = data[["Close"]].rename(columns={"Close": empresas[0]})
        else:
            precios = data

    precios = precios.dropna(axis=1, how="all")
    precios = precios.loc[:, ~precios.columns.duplicated()].sort_index(axis=1)
    Logger.info(f"Precios descargados: {precios.shape[0]} fechas x {precios.shape[1]} empresas.")
    return precios


# ==============================================================================
# SECCIÓN 3: LIMPIEZA Y PREPROCESAMIENTO DE PRECIOS
# ==============================================================================
def limpiar_precios(df: pd.DataFrame, umbral_cobertura: float = 0.80) -> pd.DataFrame:
    """
    Limpia el histórico de precios eliminando empresas con historial incompleto
    y rellenando huecos puntuales hacia adelante y atrás.
    """
    if not 0.0 < umbral_cobertura <= 1.0:
        raise ValueError("`umbral_cobertura` debe estar en el intervalo (0, 1].")

    df_limpio = df.sort_index().dropna(axis=1, how="all")
    df_limpio = df_limpio.loc[:, ~df_limpio.columns.duplicated()]
    min_dias = max(1, int(len(df_limpio) * umbral_cobertura))
    empresas_antes = df_limpio.shape[1]

    df_limpio = df_limpio.dropna(axis=1, thresh=min_dias)
    eliminadas = empresas_antes - df_limpio.shape[1]
    df_limpio = df_limpio.ffill().bfill()

    Logger.info(f"Empresas originales: {empresas_antes}")
    Logger.info(f"Empresas eliminadas (< {umbral_cobertura * 100:.0f}% datos): {eliminadas}")
    Logger.info(f"Empresas conservadas: {df_limpio.shape[1]}")
    return df_limpio


# ==============================================================================
# SECCIÓN 4: INGENIERÍA DE CARACTERÍSTICAS FINANCIERAS
# ==============================================================================
def calcular_metricas_mercado(
    precios: pd.DataFrame,
    tasa_libre_riesgo: float = 0.04,
    dias_trading: int = 252,
) -> pd.DataFrame:
    """Calcula Retorno Anual, Volatilidad Anual, Sharpe Ratio y Max Drawdown."""
    rendimientos = precios.pct_change(fill_method=None)
    metricas = pd.DataFrame(index=precios.columns)
    metricas.index.name = "Ticker"

    retorno_anual = rendimientos.mean() * dias_trading
    volatilidad_anual = rendimientos.std() * np.sqrt(dias_trading)
    volatilidad_segura = volatilidad_anual.replace(0.0, np.nan)

    metricas["Retorno_Anual"] = retorno_anual
    metricas["Volatilidad_Anual"] = volatilidad_anual
    metricas["Sharpe_Ratio"] = ((retorno_anual - tasa_libre_riesgo) / volatilidad_segura).fillna(0.0)

    acumulado = (1.0 + rendimientos.fillna(0.0)).cumprod()
    max_acumulado = acumulado.cummax().replace(0.0, np.nan)
    drawdown = ((acumulado - max_acumulado) / max_acumulado).fillna(0.0)
    metricas["Max_Drawdown"] = drawdown.min()

    Logger.info(f"Métricas de mercado calculadas para {len(metricas)} empresas.")
    return metricas


def _obtener_info_ticker(ticker: str) -> dict[str, Any]:
    """Consulta los ratios fundamentales de un ticker en Yahoo Finance."""
    try:
        info = yf.Ticker(ticker).info or {}
        return {
            "Ticker": ticker,
            "Sector": info.get("sector") or "Desconocido",
            "Deuda_Capital": info.get("debtToEquity"),
            "Liquidez_Corriente": info.get("currentRatio"),
            "ROE": info.get("returnOnEquity"),
            "ROA": info.get("returnOnAssets"),
            "Margen_Neto": info.get("profitMargins"),
            "Beta": info.get("beta"),
            "PER": info.get("trailingPE"),
        }
    except Exception:
        return {"Ticker": ticker, "Sector": "Desconocido"}


def descargar_fundamentales(tickers: list[str], max_workers: int = 10) -> pd.DataFrame:
    """Descarga de forma concurrente los ratios fundamentales de una lista de empresas."""
    if not _HAS_YFINANCE:
        raise RuntimeError("La librería `yfinance` no está instalada en el entorno.")
    if not tickers:
        raise ValueError("La lista de tickers está vacía.")

    Logger.info(f"Descargando datos fundamentales de {len(tickers)} empresas...")
    resultados: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futuros = {executor.submit(_obtener_info_ticker, t): t for t in tickers}
        for i, futuro in enumerate(as_completed(futuros), start=1):
            resultados.append(futuro.result())
            if i % 100 == 0 or i == len(tickers):
                Logger.info(f"Progreso fundamentales: {i}/{len(tickers)} empresas")

    df_fund = pd.DataFrame(resultados).set_index("Ticker")
    return df_fund[~df_fund.index.duplicated(keep="first")]


def construir_dataset_financiero(
    precios: pd.DataFrame,
    fundamentales: pd.DataFrame,
    tasa_libre_riesgo: float = 0.04,
) -> pd.DataFrame:
    """
    Une métricas de mercado y fundamentales, imputa valores faltantes con la mediana
    y aplica winsorizing [0.02, 0.02] para controlar valores atípicos extremos.
    """
    df_mercado = calcular_metricas_mercado(precios, tasa_libre_riesgo=tasa_libre_riesgo)
    df = df_mercado.join(fundamentales, how="inner")

    if "Sector" not in df.columns:
        df["Sector"] = "Desconocido"
    else:
        df["Sector"] = df["Sector"].fillna("Desconocido")

    cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in cols_num:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        mediana = df[col].median()
        df[col] = df[col].fillna(0.0 if pd.isna(mediana) else mediana)
        if df[col].nunique(dropna=True) > 1 and len(df[col]) >= 10:
            df[col] = np.asarray(winsorize(df[col].to_numpy(), limits=[0.02, 0.02]), dtype=float)

    df.index.name = "Ticker"
    Logger.info(f"Dataset financiero consolidado: {df.shape[0]} empresas x {df.shape[1]} variables.")
    return df


def generar_dataset_respaldo_sintetico(
    n_empresas: int = 120,
    semilla: int = 42,
    tasa_libre_riesgo: float = 0.04,
) -> pd.DataFrame:
    """
    Genera un dataset financiero realista de respaldo exclusivamente si no hay datos previos
    ni acceso a internet/Yahoo Finance, permitiendo validar todo el pipeline de extremo a extremo.
    """
    rng = np.random.default_rng(semilla)
    sectores = [
        "Technology",
        "Healthcare",
        "Financials",
        "Industrials",
        "Consumer Discretionary",
        "Utilities",
    ]
    tickers = [f"EMP{i:03d}" for i in range(1, n_empresas + 1)]
    volatilidad = rng.uniform(0.14, 0.55, size=n_empresas)
    sharpe_latente = rng.normal(0.65, 0.55, size=n_empresas)
    retorno = tasa_libre_riesgo + sharpe_latente * volatilidad + rng.normal(0.0, 0.03, size=n_empresas)
    sharpe = (retorno - tasa_libre_riesgo) / np.maximum(volatilidad, 1e-4)
    max_dd = -np.clip(volatilidad * rng.uniform(0.8, 1.8, size=n_empresas) + 0.08, 0.10, 0.85)
    roe = np.clip(0.14 + 0.25 * sharpe + rng.normal(0.0, 0.10, size=n_empresas), -0.35, 0.85)
    roa = np.clip(
        roe * rng.uniform(0.35, 0.70, size=n_empresas) + rng.normal(0.0, 0.02, size=n_empresas),
        -0.20,
        0.35,
    )
    margen_neto = np.clip(
        0.10 + 0.15 * roa + rng.normal(0.0, 0.06, size=n_empresas),
        -0.25,
        0.45,
    )
    deuda_capital = np.clip(rng.gamma(shape=2.0, scale=45.0, size=n_empresas), 2.0, 320.0)
    liquidez = np.clip(rng.lognormal(mean=0.45, sigma=0.45, size=n_empresas), 0.35, 5.5)
    beta = np.clip(0.45 + 1.8 * (volatilidad - 0.14) + rng.normal(0.0, 0.20, size=n_empresas), 0.2, 2.4)
    per = np.clip(18.0 + 12.0 * retorno + rng.normal(0.0, 8.0, size=n_empresas), 5.0, 75.0)

    df = pd.DataFrame(
        {
            "Retorno_Anual": retorno,
            "Volatilidad_Anual": volatilidad,
            "Sharpe_Ratio": sharpe,
            "Max_Drawdown": max_dd,
            "Deuda_Capital": deuda_capital,
            "Liquidez_Corriente": liquidez,
            "ROE": roe,
            "ROA": roa,
            "Margen_Neto": margen_neto,
            "Beta": beta,
            "PER": per,
            "Sector": rng.choice(sectores, size=n_empresas),
        },
        index=tickers,
    )
    df.index.name = "Ticker"
    Logger.warning("Se generó dataset sintético de respaldo para ejecución local/offline.")
    return df


def _features_disponibles(df: pd.DataFrame, features: list[str] | None = None) -> list[str]:
    """Devuelve las columnas numéricas válidas presentes en `df`."""
    candidatas = features if features is not None else FEATURES_FINANCIERAS
    disponibles = [col for col in candidatas if col in df.columns]
    if len(disponibles) < 2:
        raise ValueError("Se requieren al menos 2 variables numéricas disponibles para ejecutar PCA.")
    return disponibles


# ==============================================================================
# SECCIÓN 5: VALIDACIÓN DE SUPUESTOS DE PCA
# ==============================================================================
def validar_supuestos_pca(
    df: pd.DataFrame,
    features: list[str] | None = None,
    carpeta_reportes: str = "reports/figures",
) -> dict[str, Any]:
    """Evalúa correlaciones, Test de Esfericidad de Bartlett, índice KMO y gráficos de linealidad."""
    cols = _features_disponibles(df, features)
    reportes_path = Path(carpeta_reportes)
    reportes_path.mkdir(parents=True, exist_ok=True)

    X = df[cols]
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=cols, index=X.index)

    corr = X_scaled.corr()
    plt.figure(figsize=(11, 9))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Matriz de Correlación - Salud Financiera y Riesgo")
    plt.tight_layout()
    plt.savefig(reportes_path / "01_matriz_correlacion.png", dpi=300)
    plt.close()

    resultados: dict[str, Any] = {}
    if _HAS_FACTOR_ANALYZER and len(X_scaled) > len(cols):
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
            for var, val in zip(cols, kmo_por_var):
                Logger.warning(f"  - {var}: {val:.3f}")
        resultados.update(
            {
                "bartlett_chi2": float(chi2),
                "bartlett_p": float(p_value),
                "kmo_global": float(kmo_global),
                "kmo_por_variable": {var: float(val) for var, val in zip(cols, kmo_por_var)},
            }
        )
    elif not _HAS_FACTOR_ANALYZER:
        Logger.warning("`factor_analyzer` no disponible; se omite Bartlett/KMO numérico.")

    vars_clave = [
        v
        for v in ["Retorno_Anual", "Volatilidad_Anual", "ROE", "Deuda_Capital", "Beta"]
        if v in cols
    ]
    if len(vars_clave) >= 2:
        pairplot_grid = sns.pairplot(
            X_scaled[vars_clave],
            diag_kind="kde",
            plot_kws={"alpha": 0.5, "s": 25},
        )
        pairplot_grid.savefig(reportes_path / "02_linealidad_pairplot.png", dpi=200)
        plt.close("all")

    return resultados


# ==============================================================================
# SECCIÓN 6: PCA NIVEL BÁSICO - VARIANZA, SCREE PLOT, CARGAS Y HEATMAP
# ==============================================================================
def analizar_pca_basico_y_seleccion(
    df: pd.DataFrame,
    features: list[str] | None = None,
    umbral_varianza: float = 0.85,
    carpeta_reportes: str = "reports/figures",
    carpeta_tablas: str = "data/processed",
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Cubre los elementos básicos de PCA:
    1. Varianza explicada por componente (tabla CSV y gráfico de barras + acumulada).
    2. Gráfico de Scree (eigenvalores por componente y criterio de Kaiser).
    3. Cargas de los componentes principales (tabla CSV).
    4. Mapa de calor de cargas (heatmap).
    5. Selección del número óptimo de componentes según varianza acumulada.
    """
    cols = _features_disponibles(df, features)
    reportes_path = Path(carpeta_reportes)
    tablas_path = Path(carpeta_tablas)
    reportes_path.mkdir(parents=True, exist_ok=True)
    tablas_path.mkdir(parents=True, exist_ok=True)

    X_scaled = RobustScaler().fit_transform(df[cols])
    n_max = min(len(cols), max(1, len(df) - 1))
    pca_full = PCA(n_components=n_max, random_state=42)
    pca_full.fit(X_scaled)

    var_exp = pca_full.explained_variance_ratio_
    var_cum = np.cumsum(var_exp)
    eigenvalues = pca_full.explained_variance_
    nombres_pc = [f"PC{i + 1}" for i in range(n_max)]

    # 1. Varianza explicada por componente (tabla y gráfico)
    tabla_varianza = pd.DataFrame(
        {
            "Componente": nombres_pc,
            "Eigenvalor": eigenvalues,
            "Varianza_Explicada": var_exp,
            "Varianza_Explicada_Pct": var_exp * 100.0,
            "Varianza_Acumulada": var_cum,
            "Varianza_Acumulada_Pct": var_cum * 100.0,
        }
    )
    guardar_csv(tabla_varianza.set_index("Componente"), tablas_path / "varianza_explicada_pca.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    ax1.bar(
        nombres_pc,
        var_exp * 100.0,
        color="steelblue",
        alpha=0.85,
        label="Varianza Individual (%)",
    )
    ax1.plot(
        nombres_pc,
        var_cum * 100.0,
        color="crimson",
        marker="o",
        linewidth=2,
        label="Varianza Acumulada (%)",
    )
    ax1.axhline(
        umbral_varianza * 100.0,
        color="darkgreen",
        linestyle="--",
        linewidth=1.2,
        label=f"Umbral {umbral_varianza:.0%}",
    )
    ax1.set_title("1. Varianza Explicada por Componente Principal")
    ax1.set_xlabel("Componente Principal")
    ax1.set_ylabel("Porcentaje de Varianza (%)")
    ax1.set_ylim(0, 108)
    ax1.tick_params(axis="x", rotation=45)
    ax1.legend(loc="best")

    # 2. Gráfico de Scree
    ax2.plot(nombres_pc, eigenvalues, marker="s", color="darkorange", linewidth=2, label="Eigenvalor")
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, label="Criterio de Kaiser (Eigenvalor = 1)")
    ax2.set_title("2. Gráfico de Scree (Varianza / Eigenvalores)")
    ax2.set_xlabel("Componente Principal")
    ax2.set_ylabel("Eigenvalor")
    ax2.tick_params(axis="x", rotation=45)
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.savefig(reportes_path / "05_varianza_y_scree.png", dpi=300)
    plt.close()

    # 3. Cargas de los componentes (correlaciones escaladas con autovectores * sqrt(eigenvalues))
    loadings_correlacion = pd.DataFrame(
        pca_full.components_.T * np.sqrt(eigenvalues),
        columns=nombres_pc,
        index=cols,
    )
    guardar_csv(loadings_correlacion, tablas_path / "cargas_completas_pca.csv")

    # 4. Mapa de calor de cargas
    n_optimo = int(np.argmax(var_cum >= umbral_varianza) + 1)
    n_optimo = max(1, min(n_optimo, n_max))
    n_mostrar = max(1, min(max(3, n_optimo), n_max))

    plt.figure(figsize=(11, 7.5))
    sns.heatmap(
        loadings_correlacion.iloc[:, :n_mostrar],
        annot=True,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        fmt=".2f",
    )
    plt.title(f"4. Mapa de Calor de Cargas PCA (PC1 a PC{n_mostrar})")
    plt.xlabel("Componentes Principales")
    plt.ylabel("Variables Financieras")
    plt.tight_layout()
    plt.savefig(reportes_path / "06_heatmap_cargas_pca.png", dpi=300)
    plt.close()

    Logger.info(
        f"Selección automática de componentes: {n_optimo} PCs explican {var_cum[n_optimo - 1]:.2%} de la varianza."
    )
    return tabla_varianza, loadings_correlacion, n_optimo


# ==============================================================================
# SECCIÓN 7: PCA + CLUSTERING K-MEANS Y BIPLOT/PROYECCIONES
# ==============================================================================
def evaluar_numero_clusters(
    componentes: np.ndarray,
    k_max: int = 7,
    carpeta_reportes: str = "reports/figures",
) -> int:
    """Genera Método del Codo (Inercia) y Silhouette Score para elegir el K óptimo."""
    reportes_path = Path(carpeta_reportes)
    reportes_path.mkdir(parents=True, exist_ok=True)

    max_k_posible = min(k_max, len(componentes) - 1)
    if max_k_posible < 2:
        return 2

    ks = list(range(2, max_k_posible + 1))
    inercias: list[float] = []
    silhouettes: list[float] = []

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        etiquetas = km.fit_predict(componentes)
        inercias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(componentes, etiquetas)))

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
    plt.savefig(reportes_path / "03_seleccion_k_clusters.png", dpi=300)
    plt.close()
    Logger.info(f"Gráfica de selección de clusters guardada en '{reportes_path / '03_seleccion_k_clusters.png'}'")

    mejor_k = ks[int(np.argmax(silhouettes))]
    Logger.info(
        f"Mejor K según Silhouette Score en el rango evaluado: K={mejor_k} (score={max(silhouettes):.3f})"
    )
    return mejor_k


def ejecutar_pca_clustering(
    df: pd.DataFrame,
    features: list[str] | None = None,
    n_clusters: int | None = 4,
    n_components_ml: int = 6,
    carpeta_reportes: str = "reports/figures",
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Ejecuta PCA (2D para Biplot/Proyecciones + 6 componentes transformados para ML)
    y K-Means para construir el mapa de riesgo-rentabilidad, además de generar
    la gráfica dedicada de Riesgo (Volatilidad) vs. Retorno Anual.
    """
    cols = _features_disponibles(df, features)
    if len(df) < 3:
        raise ValueError("Se requieren al menos 3 observaciones para PCA y clustering.")

    reportes_path = Path(carpeta_reportes)
    reportes_path.mkdir(parents=True, exist_ok=True)

    X_scaled = RobustScaler().fit_transform(df[cols])

    pca_2d = PCA(n_components=2, random_state=42)
    componentes_2d = pca_2d.fit_transform(X_scaled)
    var_exp = pca_2d.explained_variance_ratio_
    Logger.info(
        f"Varianza explicada (2D) -> PC1: {var_exp[0]:.2%} | PC2: {var_exp[1]:.2%} | Total: {sum(var_exp):.2%}"
    )

    loadings_2d = pd.DataFrame(pca_2d.components_.T, columns=["PC1", "PC2"], index=cols)
    mejor_k = evaluar_numero_clusters(componentes_2d, k_max=7, carpeta_reportes=carpeta_reportes)

    k_elegido = mejor_k if n_clusters is None else int(n_clusters)
    k_elegido = max(2, min(k_elegido, len(df) - 1))

    kmeans = KMeans(n_clusters=k_elegido, random_state=42, n_init=10)
    df_resultado = df.copy()
    df_resultado["PC1"] = componentes_2d[:, 0]
    df_resultado["PC2"] = componentes_2d[:, 1]
    df_resultado["Cluster"] = kmeans.fit_predict(componentes_2d)

    # Gráfico Biplot + Proyecciones de los Datos + Clusters
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
    max_abs_comp = float(np.max(np.abs(componentes_2d))) if len(componentes_2d) > 0 else 1.0
    max_abs_load = float(np.max(np.abs(loadings_2d.to_numpy()))) if len(loadings_2d) > 0 else 1.0
    escala = (max_abs_comp * 0.65) / max(max_abs_load, 1e-6)

    for i, var in enumerate(cols):
        dx = float(loadings_2d.iloc[i, 0]) * escala
        dy = float(loadings_2d.iloc[i, 1]) * escala
        plt.arrow(0.0, 0.0, dx, dy, color="crimson", alpha=0.65, head_width=0.12)
        plt.text(dx * 1.12, dy * 1.12, var, color="darkred", fontsize=9, fontweight="bold")

    lim_x = float(np.percentile(np.abs(df_resultado["PC1"]), 92))
    lim_y = float(np.percentile(np.abs(df_resultado["PC2"]), 92))
    for ticker, fila in df_resultado.iterrows():
        if abs(float(fila["PC1"])) > lim_x or abs(float(fila["PC2"])) > lim_y:
            plt.text(
                float(fila["PC1"]) + 0.08,
                float(fila["PC2"]) + 0.08,
                str(ticker),
                fontsize=8,
                alpha=0.85,
            )

    plt.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    plt.axvline(0.0, color="gray", linestyle="--", linewidth=0.8)
    plt.title(
        f"Biplot y Proyecciones Riesgo-Rentabilidad (PCA + K-Means, K={k_elegido})
"
        f"Varianza Explicada (2D): {sum(var_exp):.1%}"
    )
    plt.xlabel(f"Componente Principal 1 ({var_exp[0]:.1%} varianza)")
    plt.ylabel(f"Componente Principal 2 ({var_exp[1]:.1%} varianza)")
    plt.tight_layout()
    plt.savefig(reportes_path / "04_biplot_pca_clusters.png", dpi=300)
    plt.close()

    # Gráfica exclusiva: Riesgo (Volatilidad) vs. Retorno Anual
    if "Volatilidad_Anual" in df_resultado.columns and "Retorno_Anual" in df_resultado.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df_resultado,
            x="Volatilidad_Anual",
            y="Retorno_Anual",
            hue="Cluster",
            palette="viridis",
            s=85,
            alpha=0.85,
        )
        plt.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
        plt.title("Relación Riesgo (Volatilidad Anual) vs. Retorno Anual")
        plt.xlabel("Riesgo (Volatilidad Anual)")
        plt.ylabel("Retorno Anual")
        plt.tight_layout()
        plt.savefig(reportes_path / "04b_riesgo_vs_retorno.png", dpi=300)
        plt.close()
        Logger.info(f"Gráfica Riesgo vs. Retorno guardada en '{reportes_path / '04b_riesgo_vs_retorno.png'}'")

    # Extracción de 6 componentes principales para Machine Learning
    n_comp_validos = min(n_components_ml, X_scaled.shape[1], X_scaled.shape[0] - 1)
    pca_ml = PCA(n_components=n_comp_validos, random_state=42)
    componentes_ml = pca_ml.fit_transform(X_scaled)
    for idx in range(n_comp_validos):
        df_resultado[f"PC{idx + 1}"] = componentes_ml[:, idx]

    Logger.info(
        f"PCA para ML ({n_comp_validos} componentes): "
        f"{np.sum(pca_ml.explained_variance_ratio_):.2%} de varianza acumulada."
    )
    return df_resultado, loadings_2d, componentes_ml


# ==============================================================================
# SECCIÓN 8: MÓDULO DE MACHINE LEARNING (6 COMPONENTES VS. ORIGINALES)
# ==============================================================================
def entrenar_y_evaluar_modelos_ml(
    df_resultado: pd.DataFrame,
    componentes_pca: np.ndarray,
    features: list[str] | None = None,
    carpeta_reportes: str = "reports/figures",
    carpeta_tablas: str = "data/processed",
) -> pd.DataFrame:
    """
    Entrena modelos de clasificación (Regresión Logística, Random Forest y SVM)
    para predecir empresas de alto desempeño (Sharpe Ratio superior a la mediana),
    comparando el rendimiento con 6 componentes principales vs. variables originales.
    """
    cols = _features_disponibles(df_resultado, features)
    reportes_path = Path(carpeta_reportes)
    tablas_path = Path(carpeta_tablas)
    reportes_path.mkdir(parents=True, exist_ok=True)
    tablas_path.mkdir(parents=True, exist_ok=True)

    # Variable objetivo binaria: Salud/Rendimiento ajustado por riesgo superior a la mediana
    if "Sharpe_Ratio" in df_resultado.columns:
        y = (df_resultado["Sharpe_Ratio"] > df_resultado["Sharpe_Ratio"].median()).astype(int).to_numpy()
        features_orig = [c for c in cols if c != "Sharpe_Ratio"]
    else:
        y = (df_resultado["Retorno_Anual"] > df_resultado["Retorno_Anual"].median()).astype(int).to_numpy()
        features_orig = [c for c in cols if c != "Retorno_Anual"]

    X_orig = RobustScaler().fit_transform(df_resultado[features_orig])
    X_pca = componentes_pca
    n_pcs = X_pca.shape[1]

    modelos = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM_RBF": SVC(probability=True, random_state=42),
    }

    conjuntos = {
        f"PCA ({n_pcs} componentes)": X_pca,
        f"Originales ({len(features_orig)} vars)": X_orig,
    }

    registros: list[dict[str, Any]] = []
    matrices_confusion: dict[str, np.ndarray] = {}

    for nombre_set, X_data in conjuntos.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y, test_size=0.25, random_state=42, stratify=y
        )
        for nombre_modelo, modelo in modelos.items():
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            y_prob = modelo.predict_proba(X_test)[:, 1]

            registros.append(
                {
                    "Representacion": nombre_set,
                    "Modelo": nombre_modelo,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, zero_division=0),
                    "Recall": recall_score(y_test, y_pred, zero_division=0),
                    "F1_Score": f1_score(y_test, y_pred, zero_division=0),
                    "ROC_AUC": roc_auc_score(y_test, y_prob),
                }
            )
            matrices_confusion[f"{nombre_modelo}
({nombre_set})"] = confusion_matrix(y_test, y_pred)

    df_metricas = pd.DataFrame(registros)
    guardar_csv(df_metricas, tablas_path / "comparativa_ml_pca_vs_original.csv", index=False)

    # Gráfica comparativa de F1-Score y ROC-AUC
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(data=df_metricas, x="Modelo", y="F1_Score", hue="Representacion", ax=ax1, palette="Set2")
    ax1.set_title(f"Comparativa F1-Score: PCA ({n_pcs} PCs) vs. Originales")
    ax1.set_ylim(0, 1.05)

    sns.barplot(data=df_metricas, x="Modelo", y="ROC_AUC", hue="Representacion", ax=ax2, palette="Set2")
    ax2.set_title(f"Comparativa ROC-AUC: PCA ({n_pcs} PCs) vs. Originales")
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(reportes_path / "07_comparativa_ml_pca_vs_original.png", dpi=300)
    plt.close()

    # Matrices de confusión para los modelos entrenados con 6 PCs
    claves_pca = [k for k in matrices_confusion if f"PCA ({n_pcs}" in k]
    if claves_pca:
        fig, axes = plt.subplots(1, len(claves_pca), figsize=(5 * len(claves_pca), 4.2))
        if len(claves_pca) == 1:
            axes = [axes]
        for ax, clave in zip(axes, claves_pca):
            sns.heatmap(matrices_confusion[clave], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set_title(clave, fontsize=10)
            ax.set_xlabel("Predicción")
            ax.set_ylabel("Real")
        plt.tight_layout()
        plt.savefig(reportes_path / "08_matrices_confusion_pca6.png", dpi=300)
        plt.close()

    Logger.info("Módulo de Machine Learning finalizado y métricas guardadas.")
    return df_metricas


# ==============================================================================
# SECCIÓN 9: FLUJO PRINCIPAL DE EJECUCIÓN
# ==============================================================================
def main() -> None:
    ruta_dataset = Path("data/processed/dataset_financiero_sp500.csv")
    ruta_precios = Path("data/raw/precios_sp500.csv")
    ruta_fund = Path("data/raw/fundamentales_sp500.csv")

    if ruta_dataset.exists():
        Logger.info(f"Cargando dataset procesado existente desde '{ruta_dataset}'...")
        df_financiero = pd.read_csv(ruta_dataset, index_col="Ticker")
    else:
        try:
            if ruta_precios.exists() and ruta_fund.exists():
                precios_limpios = pd.read_csv(ruta_precios, index_col=0, parse_dates=True)
                df_fund = pd.read_csv(ruta_fund, index_col="Ticker")
            else:
                tickers = obtener_tickers_sp500()
                precios_raw = descargar_precios(tickers, periodo="5y")
                precios_limpios = limpiar_precios(precios_raw, umbral_cobertura=0.80)
                guardar_csv(precios_limpios, ruta_precios)

                df_fund = descargar_fundamentales(precios_limpios.columns.tolist())
                guardar_csv(df_fund, ruta_fund)

            df_financiero = construir_dataset_financiero(precios_limpios, df_fund)
            guardar_csv(df_financiero, ruta_dataset)
        except Exception as exc:
            Logger.warning(f"No se pudo completar la descarga en línea ({exc}). Usando respaldo local.")
            df_financiero = generar_dataset_respaldo_sintetico(n_empresas=120)
            guardar_csv(df_financiero, ruta_dataset)

    validar_supuestos_pca(df_financiero)
    analizar_pca_basico_y_seleccion(df_financiero, umbral_varianza=0.85)

    # Ejecuta clustering y extrae 6 componentes principales para ML + gráfica Riesgo vs Retorno
    df_resultado, _, componentes_pca_6 = ejecutar_pca_clustering(
        df_financiero,
        n_clusters=4,
        n_components_ml=6,
    )
    guardar_csv(df_resultado, "data/processed/resultados_pca_clustering.csv")

    # Entrena y evalúa Machine Learning con los 6 componentes principales
    df_metricas_ml = entrenar_y_evaluar_modelos_ml(df_resultado, componentes_pca_6)
    print("\n=== Resumen de Métricas ML (6 Componentes Principales vs. Variables Originales) ===")
    print(df_metricas_ml.to_string(index=False))


if __name__ == "__main__":
    main()
