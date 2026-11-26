import numpy as np
import pandas as pd

df = pd.read_csv("data/processed/resultados_clusters.csv", index_col=0)
ret_med = df["Retorno_Anual"].median()
vol_med = df["Volatilidad_Anual"].median()

df["Cuadrante"] = np.where(
    df["Retorno_Anual"] > ret_med,
    np.where(df["Volatilidad_Anual"] > vol_med, "AltoRet_AltaVol", "AltoRet_BajaVol"),
    np.where(df["Volatilidad_Anual"] > vol_med, "BajoRet_AltaVol", "BajoRet_BajaVol"),
)
tabla = pd.crosstab(df["Cuadrante"], df["Cluster"])
print(tabla)
print("\nPorcentaje por fila (%):")
print((tabla.div(tabla.sum(axis=1), axis=0) * 100).round(1))