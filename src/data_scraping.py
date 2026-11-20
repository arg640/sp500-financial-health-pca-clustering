import time
from pathlib import Path

import pandas as pd
import yfinance as yf
from yftickers import YahooFinanceTickers
"""
Se descarga un data frame de precios de mercado para un conjunto de empresas del S&P 500 usando la librería yfinance.
se utiliza el método yf.download() para obtener datos históricos de precios de mercado para un conjunto de empresas del S&P 500. 
El período de tiempo se establece, los datos se obtienen por el símbolo de la empresa.

Las columnas del dataframe resultante contienen: Open (precio de la accion al inicio de la sesión), 
High (precio más alto alcanzado durante la sesión), Low (precio más bajo alcanzado durante la sesión), 
Close (precio de cierre de la sesión) y Volume (número de acciones negociadas durante la sesión).

Dado que no se pueden descargar todos los datos de una sola vez, se hace un bucle para descargar en partes
luego se unen en un solo dataframe.
"""
paso = 50
periodo = "5y"
espera = 2

def descargar_precios_empresas(empresas, paso=paso, periodo=periodo, espera=espera):
    """
    Descarga los precios de mercado para un conjunto de empresas del S&P 500.

    Args:
        empresas (list): Lista de símbolos de las empresas.
        paso (int): Número de empresas a descargar en cada iteración.
        periodo (str): Período de tiempo para los datos históricos.
        espera (int): Tiempo de espera entre descargas en segundos.

    Returns:
        pd.DataFrame: DataFrame con los precios de mercado descargados.
    """
    precios = pd.DataFrame()
    for i in range(0, len(empresas), paso):
        sub_empresas = empresas[i:i + paso]
        print(f"Descargando precios para: {sub_empresas}")
        data = yf.download(sub_empresas, period=periodo, auto_adjust=True, group_by="ticker", progress=False)
        precios = pd.concat([precios, data], axis=1)
        time.sleep(espera)  
    precios = precios.loc[:, ~precios.columns.duplicated()]
    return precios

def guardar_precios_en_csv(precios, ruta_archivo):
    """
    Guarda los precios de mercado en un archivo CSV.

    Args:
        precios (pd.DataFrame): DataFrame con los precios de mercado.
        ruta_archivo (str): Ruta del archivo CSV donde se guardarán los datos.
    """
    Path(ruta_archivo).parent.mkdir(parents=True, exist_ok=True)  
    precios.to_csv(ruta_archivo)
    print(f"Precios guardados en {ruta_archivo}")
    

def get_sp500_tickers_yftickers():
    yft = YahooFinanceTickers()
    tickers = yft.get_tickers("SP500")
    return [
        t.strip().upper().replace(".", "-")
        for t in tickers
        if isinstance(t, str) and "." not in t.strip()
    ]

def main():
    empresas = get_sp500_tickers_yftickers()   
    print("Empresas:", len(empresas))
    precios = descargar_precios_empresas(empresas)
    guardar_precios_en_csv(precios, "data/raw/precios_empresas.csv")
    
if __name__ == "__main__":
    main()
