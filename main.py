from pathlib import Path
import pandas as pd
from utils.io_helpers import guardar_csv
from utils.logger import Logger
from src.data_scraping import obtener_tickers_sp500, descargar_precios
from src.data_explorer import limpiar_precios
from src.feature_engineering import descargar_fundamentales, construir_dataset_financiero
from src.pca_validation import validar_supuestos_pca
from src.pca_clustering import ejecutar_pca_clustering


def main() -> None:
    """Orquesta el pipeline completo: descarga, limpieza, métricas, validación PCA y clustering."""
    ruta_raw = "data/raw/precios_empresas.csv"
    ruta_clean = "data/processed/precios_limpios.csv"
    ruta_fund = "data/raw/fundamentales_empresas.csv"
    ruta_dataset = "data/processed/dataset_financiero.csv"
    if not Path(ruta_raw).exists():
        tickers = obtener_tickers_sp500()
        df_raw = descargar_precios(tickers, periodo="5y")
        guardar_csv(df_raw, ruta_raw)
    else:
        Logger.info(f"Usando precios existentes en '{ruta_raw}'")
        df_raw = pd.read_csv(ruta_raw, index_col="Date", parse_dates=True)
    df_clean = limpiar_precios(df_raw, umbral_cobertura=0.80)
    guardar_csv(df_clean, ruta_clean)
    if not Path(ruta_fund).exists():
        df_fund = descargar_fundamentales(df_clean.columns.tolist(), max_workers=10)
        guardar_csv(df_fund, ruta_fund)
    else:
        Logger.info(f"Usando fundamentales existentes en '{ruta_fund}'")
        df_fund = pd.read_csv(ruta_fund, index_col="Ticker")
    df_financiero = construir_dataset_financiero(df_clean, df_fund)
    guardar_csv(df_financiero, ruta_dataset)
    features = [
        "Retorno_Anual", "Volatilidad_Anual", "Sharpe_Ratio", "Max_Drawdown",
        "Deuda_Capital", "Liquidez_Corriente", "ROE", "ROA", "Margen_Neto", "Beta", "PER",
    ]
    validar_supuestos_pca(df_financiero, features, carpeta_reportes="reports/figures")
    df_clusters, loadings = ejecutar_pca_clustering(
        df_financiero, features, n_clusters=4, carpeta_reportes="reports/figures"
    )
    guardar_csv(df_clusters, "data/processed/resultados_clusters.csv")
    guardar_csv(loadings, "data/processed/cargas_pca.csv")
    resumen = df_clusters.groupby("Cluster")[["Retorno_Anual", "Volatilidad_Anual", "ROE", "Deuda_Capital", "Beta"]].mean()
    Logger.info(f"\n--- PERFIL PROMEDIO POR CLUSTER ---\n{resumen.round(3)}")
if __name__ == "__main__":
    main()