from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats.mstats import winsorize
from utils.logger import Logger


def calcular_metricas_mercado(precios: pd.DataFrame, tasa_libre_riesgo: float = 0.04) -> pd.DataFrame:
    """
    Calcula indicadores de rentabilidad y riesgo de mercado a partir de precios históricos.

    Args:
        precios (pd.DataFrame): DataFrame limpio de precios de cierre ajustados.
        tasa_libre_riesgo (float): Tasa anual libre de riesgo para el Ratio de Sharpe. Defaults to 0.04.

    Returns:
        pd.DataFrame: Métricas anualizadas (Retorno, Volatilidad, Sharpe y Max Drawdown) por ticker.
    """
    rendimientos = precios.pct_change(fill_method=None)
    dias_trading = 252
    metricas = pd.DataFrame(index=precios.columns)
    metricas["Retorno_Anual"] = rendimientos.mean() * dias_trading
    metricas["Volatilidad_Anual"] = rendimientos.std() * np.sqrt(dias_trading)
    metricas["Sharpe_Ratio"] = (metricas["Retorno_Anual"] - tasa_libre_riesgo) / metricas["Volatilidad_Anual"]
    acumulado = (1 + rendimientos.fillna(0)).cumprod()
    max_acumulado = acumulado.cummax()
    drawdown = (acumulado - max_acumulado) / max_acumulado
    metricas["Max_Drawdown"] = drawdown.min()
    Logger.info(f"Métricas de mercado calculadas para {len(metricas)} empresas.")
    return metricas

def _obtener_info_ticker(ticker: str) -> dict:
    """
    Consulta los ratios fundamentales de un solo ticker en Yahoo Finance.

    Args:
        ticker (str): Símbolo bursátil de la empresa.

    Returns:
        dict: Diccionario con sector y ratios financieros clave.
    """
    try:
        info = yf.Ticker(ticker).info
        return {
            "Ticker": ticker,
            "Sector": info.get("sector", "Desconocido"),
            "Deuda_Capital": info.get("debtToEquity"),
            "Liquidez_Corriente": info.get("currentRatio"),
            "ROE": info.get("returnOnEquity"),
            "ROA": info.get("returnOnAssets"),
            "Margen_Neto": info.get("profitMargins"),
            "Beta": info.get("beta"),
            "PER": info.get("trailingPE"),
        }
    except Exception:
        return {"Ticker": ticker}

def descargar_fundamentales(tickers: list[str], max_workers: int = 10) -> pd.DataFrame:
    """
    Descarga de forma concurrente los ratios de salud financiera de una lista de empresas.

    Args:
        tickers (list[str]): Lista de símbolos bursátiles.
        max_workers (int): Número de hilos simultáneos para las peticiones. Defaults to 10.

    Returns:
        pd.DataFrame: DataFrame indexado por Ticker con las métricas fundamentales.
    """
    Logger.info(f"Descargando datos fundamentales de {len(tickers)} empresas...")
    resultados = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futuros = {executor.submit(_obtener_info_ticker, t): t for t in tickers}
        for i, futuro in enumerate(as_completed(futuros), start=1):
            resultados.append(futuro.result())
            if i % 100 == 0 or i == len(tickers):
                Logger.info(f"Progreso fundamentales: {i}/{len(tickers)} empresas")
    return pd.DataFrame(resultados).set_index("Ticker")


def construir_dataset_financiero(precios: pd.DataFrame, fundamentales: pd.DataFrame) -> pd.DataFrame:
    """
    Une métricas de mercado y fundamentales, imputa nulos con la mediana y acota outliers extremos.

    Args:
        precios (pd.DataFrame): DataFrame limpio de precios históricos.
        fundamentales (pd.DataFrame): DataFrame con ratios fundamentales por empresa.

    Returns:
        pd.DataFrame: Matriz final (Empresas x Indicadores) lista para análisis multivariado.
    """
    df_mercado = calcular_metricas_mercado(precios)
    df = df_mercado.join(fundamentales, how="inner")
    cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
    df[cols_num] = df[cols_num].replace([np.inf, -np.inf], np.nan)
    df[cols_num] = df[cols_num].fillna(df[cols_num].median())
    for col in cols_num:
        df[col] = np.asarray(winsorize(df[col], limits=[0.02, 0.02]))
    Logger.info(f"Dataset financiero consolidado: {df.shape[0]} empresas x {df.shape[1]} variables.")
    return df