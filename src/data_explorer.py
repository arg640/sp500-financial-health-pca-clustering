import pandas as pd
from utils.logger import Logger


def limpiar_precios(df: pd.DataFrame, umbral_cobertura: float = 0.80) -> pd.DataFrame:
    """
    Limpia el histórico de precios eliminando empresas con historial incompleto
    y rellenando huecos puntuales sin borrar filas del calendario bursátil.

    Args:
        df (pd.DataFrame): DataFrame crudo con fechas en el índice y tickers en columnas.
        umbral_cobertura (float): Porcentaje mínimo de días cotizados para conservar la empresa. Defaults to 0.80.

    Returns:
        pd.DataFrame: DataFrame filtrado e imputado listo para calcular métricas.
    """
    df = df.sort_index().dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.duplicated()]
    min_dias = int(len(df) * umbral_cobertura)
    empresas_antes = df.shape[1]
    df = df.dropna(axis=1, thresh=min_dias)
    eliminadas = empresas_antes - df.shape[1]
    df = df.ffill()
    Logger.info(f"Empresas originales: {empresas_antes}")
    Logger.info(f"Empresas eliminadas (< {umbral_cobertura * 100:.0f}% datos): {eliminadas}")
    Logger.info(f"Empresas conservadas: {df.shape[1]}")
    return df