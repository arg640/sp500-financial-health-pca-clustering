from pathlib import Path
import pandas as pd
from utils.logger import Logger


def guardar_csv(df: pd.DataFrame, ruta: str) -> None:
    """
    Crea el directorio si no existe y guarda el DataFrame en CSV
    Args:
        df (pd.DataFrame): El dataframe a guardar
        ruta (str): Ruta en donde quedara guardado el dataframe
    """
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta)
    Logger.info(f"Archivo guardado en '{ruta}' ({df.shape[0]} filas x {df.shape[1]} columnas)")