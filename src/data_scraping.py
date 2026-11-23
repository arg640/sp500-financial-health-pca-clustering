import pandas as pd
import yfinance as yf
from utils.logger import Logger


def obtener_tickers_sp500() -> list[str]:
    """
    Obtiene y normaliza la lista vigente de tickers del S&P 500.

    Returns:
        list[str]: Una lista ordenada con todos los tickers vigentes del S&P 500.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(url, storage_options={"User-Agent": "Mozilla/5.0"})[0]
    tickers = df["Symbol"].str.strip().str.upper().str.replace(".", "-", regex=False)
    return sorted(tickers.unique().tolist())


def descargar_precios(empresas: list[str], periodo: str = "5y") -> pd.DataFrame:
    """
    Descarga precios de cierre ajustados y elimina columnas vacías o duplicadas.

    Args:
        empresas (list[str]): Lista de empresas de las cuales se van a descargar datos.
        periodo (str): El plazo del cual se quieren descargar datos. Defaults to "5y".

    Returns:
        pd.DataFrame: Un DataFrame con los precios históricos limpios de columnas vacías.
    """
    Logger.info(f"Descargando histórico ({periodo}) de {len(empresas)} empresas")
    data = yf.download(
        empresas,
        period=periodo,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    precios = data.xs("Close", axis=1, level=1).dropna(axis=1, how="all")
    return precios.loc[:, ~precios.columns.duplicated()].sort_index(axis=1)