import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

FEATURES = ["Retorno_Anual", "Volatilidad_Anual", "Sharpe_Ratio", "Max_Drawdown",
            "Deuda_Capital", "Liquidez_Corriente", "ROE", "ROA", "Margen_Neto", "Beta", "PER"]

Path("reports/figures").mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/processed/resultados_clusters.csv", index_col=0)
print(f"Dataset: {df.shape[0]} empresas, columnas: {df.columns.tolist()}\n")

# ============================================================
# 1. CROSSTAB CUADRANTE RIESGO-RETORNO vs CLUSTER
# ============================================================
ret_med = df["Retorno_Anual"].median()
vol_med = df["Volatilidad_Anual"].median()

df["Cuadrante"] = np.where(
    df["Retorno_Anual"] > ret_med,
    np.where(df["Volatilidad_Anual"] > vol_med, "AltoRet_AltaVol", "AltoRet_BajaVol"),
    np.where(df["Volatilidad_Anual"] > vol_med, "BajoRet_AltaVol", "BajoRet_BajaVol"),
)

print("=" * 70)
print("CROSSTAB: Cuadrante riesgo-retorno vs Cluster")
print("=" * 70)
tabla = pd.crosstab(df["Cuadrante"], df["Cluster"])
print(tabla)
print("\nPorcentaje por fila (%):")
print((tabla.div(tabla.sum(axis=1), axis=0) * 100).round(1))

# ============================================================
# 2. HEATMAP Z-SCORES POR CLUSTER (Fig. principal del paper)
# ============================================================
z = df.copy()
z[FEATURES] = (z[FEATURES] - z[FEATURES].mean()) / z[FEATURES].std()

perfil_z = z.groupby("Cluster")[FEATURES].mean().T

plt.figure(figsize=(10, 8))
sns.heatmap(perfil_z, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-2, vmax=2, cbar_kws={"label": "Z-score (vs. media del dataset)"})
plt.title("Perfil financiero por cluster (Z-scores sobre las 11 variables)")
plt.xlabel("Cluster")
plt.ylabel("Variable financiera")
plt.tight_layout()
plt.savefig("reports/figures/07_heatmap_zscore_clusters.png", dpi=300)
plt.close()
print("\n[OK] Heatmap Z-scores guardado en 'reports/figures/07_heatmap_zscore_clusters.png'")

# ============================================================
# 3. COMPOSICIÓN SECTORIAL (si existe columna Sector)
# ============================================================
if "Sector" in df.columns:
    print("\n" + "=" * 70)
    print("COMPOSICIÓN SECTORIAL POR CLUSTER (conteo)")
    print("=" * 70)
    comp = pd.crosstab(df["Cluster"], df["Sector"])
    print(comp)
    print("\nPorcentaje por cluster (%):")
    print((comp.div(comp.sum(axis=1), axis=0) * 100).round(1))

    plt.figure(figsize=(12, 6))
    (comp.div(comp.sum(axis=1), axis=0) * 100).plot(
        kind="bar", stacked=True, figsize=(12, 6), colormap="tab20"
    )
    plt.title("Composición sectorial por cluster (%)")
    plt.xlabel("Cluster")
    plt.ylabel("Porcentaje (%)")
    plt.legend(title="Sector", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig("reports/figures/08_composicion_sectorial.png", dpi=300)
    plt.close()
    print("\n[OK] Composición sectorial guardada en 'reports/figures/08_composicion_sectorial.png'")
else:
    print("\n[!] No hay columna 'Sector' en el CSV; se omite composición sectorial.")

# ============================================================
# 4. RESUMEN DE TAMAÑOS
# ============================================================
print("\n" + "=" * 70)
print("TAMAÑOS DE CLUSTER")
print("=" * 70)
print(df["Cluster"].value_counts().sort_index())