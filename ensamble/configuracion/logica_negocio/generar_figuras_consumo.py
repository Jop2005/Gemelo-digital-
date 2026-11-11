"""
Genera las figuras del Capítulo 3 para el dominio de consumo.

Figuras:
    1. fig_metricas_globales_consumo.png    → MAE, RMSE, R², MAPE, MASE
    2. fig_metricas_segmento_consumo.png    → MAE por segmento (RF vs XGB)
    3. fig_importancia_rf_consumo.png       → Importancia de features (RF)
    4. fig_importancia_xgb_consumo.png      → Importancia de features (XGB)
    5. fig_pesos_ensamble_consumo.png       → Pesos del ensamble por segmento

Uso:
    python -m configuracion.logica_negocio.generar_figuras_consumo
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = Path(__file__).parent.parent.parent
sys.path.insert(0, str(RAIZ))
os.chdir(RAIZ)

CARPETA_FIGURAS = RAIZ / 'artifacts' / 'figuras' / 'consumo'
CARPETA_FIGURAS.mkdir(parents=True, exist_ok=True)

COLOR_RF = "#2A7DB5"
COLOR_XGB = "#9B2A6B"

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ============================================================
# DATOS
# ============================================================

METRICAS_GLOBALES = {
    "Random Forest": {"MAE": 13.31, "RMSE": 17.40, "R2": 0.946, "MAPE": 5.63, "MASE": 0.7708},
    "XGBoost":       {"MAE": 13.39, "RMSE": 17.50, "R2": 0.945, "MAPE": 5.67, "MASE": 0.7755},
}

MAE_SEGMENTO = {
    "Random Forest": [11.21, 15.28, 15.73, 13.44, 11.11, 10.93, 12.39, 12.51],
    "XGBoost":       [11.11, 15.57, 15.99, 13.57, 11.11, 10.73, 12.39, 12.51],
}

IMPORTANCIA_RF = {"Mes": 68.05, "Hora": 25.16, "tipo_dia_bin": 6.79}
IMPORTANCIA_XGB = {"Mes": 76.0, "Hora": 9.7, "tipo_dia_bin": 14.3}

PESOS_ENSAMBLE = {
    "Random Forest": [1.0, 0.750, 1.0, 0.750, 1.0, 0.250, 0.750, 1.0],
    "XGBoost":       [0.0, 0.250, 0.0, 0.250, 0.0, 0.750, 0.250, 0.0],
}

SEGMENTOS = list(range(8))
ETIQUETAS_SEG = ["0\nMad. Lab", "1\nMañ. Lab", "2\nTarde Lab", "3\nNoche Lab",
                 "4\nMad. Finde", "5\nMañ. Finde", "6\nTarde Finde", "7\nNoche Finde"]


# ============================================================
# FIGURA 1 — Métricas globales
# ============================================================

def fig_metricas_globales():
    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    modelos = ["Random Forest", "XGBoost"]
    colores = [COLOR_RF, COLOR_XGB]
    met = ["MAE", "RMSE", "R2", "MAPE", "MASE"]
    ylabels = ["MAE (kW)", "RMSE (kW)", "R²", "MAPE (%)", "MASE"]

    for ax, m, yl in zip(axes, met, ylabels):
        vals = [METRICAS_GLOBALES[mod][m] for mod in modelos]
        bars = ax.bar(modelos, vals, color=colores, edgecolor="white", width=0.7)
        best = min(vals) if m != "R2" else max(vals)
        for b, v in zip(bars, vals):
            if v == best:
                b.set_edgecolor("#D62728")
                b.set_linewidth(2.5)
            ax.text(b.get_x() + b.get_width()/2, b.get_height()*1.02,
                    f"{v:.4g}" if m in ("R2", "MASE") else f"{v:.2f}",
                    ha="center", fontsize=9, fontweight="bold")
        ax.set_title(yl, fontweight="bold")
        ax.set_ylim(0, max(vals)*1.25 if max(vals) > 0 else 1)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticks(range(len(modelos)))
        ax.set_xticklabels(["RF", "XGB"])

    fig.suptitle("Métricas globales — Consumo", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    ruta = CARPETA_FIGURAS / "fig_metricas_globales.png"
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    print(f"✓ {ruta}")


# ============================================================
# FIGURA 2 — MAE por segmento
# ============================================================

def fig_mae_segmento():
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(SEGMENTOS))
    w = 0.35
    rf = MAE_SEGMENTO["Random Forest"]
    xg = MAE_SEGMENTO["XGBoost"]

    ax.bar(x - w/2, rf, w, label="Random Forest", color=COLOR_RF, edgecolor="white")
    ax.bar(x + w/2, xg, w, label="XGBoost", color=COLOR_XGB, edgecolor="white")

    for i, (v1, v2) in enumerate(zip(rf, xg)):
        ax.text(i - w/2, v1 + 0.1, f"{v1:.2f}", ha="center", fontsize=8)
        ax.text(i + w/2, v2 + 0.1, f"{v2:.2f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(ETIQUETAS_SEG, fontsize=8)
    ax.set_ylabel("MAE (kW)")
    ax.set_title("MAE por segmento — Consumo", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    ruta = CARPETA_FIGURAS / "fig_mae_segmento.png"
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    print(f"✓ {ruta}")


# ============================================================
# FIGURA 3 — Importancia de características (RF)
# ============================================================

def fig_importancia_rf():
    fig, ax = plt.subplots(figsize=(8, 4))
    items = sorted(IMPORTANCIA_RF.items(), key=lambda x: x[1])
    nombres = [k for k, _ in items]
    valores = [v for _, v in items]

    bars = ax.barh(nombres, valores, color=COLOR_RF, edgecolor="white")
    for b, v in zip(bars, valores):
        ax.text(v + 0.5, b.get_y() + b.get_height()/2, f"{v:.2f}%", va="center", fontsize=9)

    ax.set_xlabel("Importancia (%)")
    ax.set_title("Importancia de características — Random Forest (Consumo)", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    ruta = CARPETA_FIGURAS / "fig_importancia_rf.png"
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    print(f"✓ {ruta}")


# ============================================================
# FIGURA 4 — Importancia de características (XGB)
# ============================================================

def fig_importancia_xgb():
    fig, ax = plt.subplots(figsize=(8, 4))
    items = sorted(IMPORTANCIA_XGB.items(), key=lambda x: x[1])
    nombres = [k for k, _ in items]
    valores = [v for _, v in items]

    bars = ax.barh(nombres, valores, color=COLOR_XGB, edgecolor="white")
    for b, v in zip(bars, valores):
        ax.text(v + 0.5, b.get_y() + b.get_height()/2, f"{v:.2f}%", va="center", fontsize=9)

    ax.set_xlabel("Importancia (%)")
    ax.set_title("Importancia de características — XGBoost (Consumo)", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    ruta = CARPETA_FIGURAS / "fig_importancia_xgb.png"
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    print(f"✓ {ruta}")


# ============================================================
# FIGURA 5 — Pesos del ensamble por segmento
# ============================================================

def fig_pesos_ensamble():
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(SEGMENTOS))
    rf = np.array(PESOS_ENSAMBLE["Random Forest"])
    xg = np.array(PESOS_ENSAMBLE["XGBoost"])

    ax.bar(x, rf, label="Random Forest", color=COLOR_RF, edgecolor="white")
    ax.bar(x, xg, bottom=rf, label="XGBoost", color=COLOR_XGB, edgecolor="white")

    for i, (v1, v2) in enumerate(zip(rf, xg)):
        if v1 > 0.05:
            ax.text(i, v1/2, f"{v1*100:.0f}%", ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold")
        if v2 > 0.05:
            ax.text(i, v1 + v2/2, f"{v2*100:.0f}%", ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(ETIQUETAS_SEG, fontsize=8)
    ax.set_ylabel("Peso")
    ax.set_ylim(0, 1.05)
    ax.set_title("Pesos del ensamblaje por segmento — Consumo", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    ruta = CARPETA_FIGURAS / "fig_pesos_ensamble.png"
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    print(f"✓ {ruta}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  GENERANDO FIGURAS — CONSUMO")
    print("=" * 60)
    print(f"  Carpeta de salida: {CARPETA_FIGURAS}\n")

    fig_metricas_globales()
    fig_mae_segmento()
    fig_importancia_rf()
    fig_importancia_xgb()
    fig_pesos_ensamble()

    print("\n✓ Todas las figuras generadas correctamente")
    print("=" * 60)