"""
Genera las figuras del Capítulo 3 para el dominio de generación.

Figuras:
    1. fig_3_metricas_generacion.png       → MAE, RMSE, R², MAPE, MASE globales
    2. fig_3_metricas_diurnas_generacion.png → R², nRMSE, nMAE, MASE diurnos
    3. fig_3_mae_segmento_generacion.png    → MAE por segmento (RF vs XGB)
    4. fig_3_r2_segmento_generacion.png     → R² por segmento (RF vs XGB)
    5. fig_3_importancia_rf_generacion.png  → Importancia de features (RF)
    6. fig_3_importancia_xgb_generacion.png → Importancia de features (XGB)
    7. fig_3_pesos_ensamble_generacion.png  → Pesos del ensamble por segmento

Uso:
    python -m configuracion.logica_negocio.generar_figuras_generacion
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Asegurar imports desde la raíz
RAIZ = Path(__file__).parent.parent.parent
sys.path.insert(0, str(RAIZ))
os.chdir(RAIZ)


# ============================================================
# CONFIGURACIÓN
# ============================================================

CARPETA_FIGURAS = RAIZ / 'artifacts' / 'figuras' / 'generacion'
CARPETA_FIGURAS.mkdir(parents=True, exist_ok=True)

# Colores consistentes
COLOR_RF = "#2A7DB5"
COLOR_XGB = "#9B2A6B"

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


# ============================================================
# DATOS (extraídos del pipeline)
# ============================================================

# Métricas globales
METRICAS_GLOBALES = {
    "Random Forest": {"MAE": 0.02, "RMSE": 0.05, "R2": 0.9575, "MAPE": 26.35, "MASE": 0.4294},
    "XGBoost":       {"MAE": 0.02, "RMSE": 0.05, "R2": 0.9562, "MAPE": 34.03, "MASE": 0.4370},
}

# Métricas diurnas
METRICAS_DIURNAS = {
    "Random Forest": {"R2": 0.9025, "nRMSE": 7.25, "nMAE": 4.72, "MASE": 0.8597},
    "XGBoost":       {"R2": 0.8996, "nRMSE": 7.35, "nMAE": 4.74, "MASE": 0.8639},
}

# MAE por segmento
MAE_SEGMENTO = {
    "Random Forest": [0.00, 0.03, 0.05, 0.06, 0.06, 0.06],
    "XGBoost":       [0.00, 0.03, 0.05, 0.06, 0.06, 0.06],
}

# R² por segmento
R2_SEGMENTO = {
    "Random Forest": [0.9405, 0.8894, 0.8877, 0.8413, 0.7596, 0.5432],
    "XGBoost":       [0.9373, 0.8826, 0.8876, 0.8365, 0.7500, 0.5333],
}

# Importancia de características
IMPORTANCIA_RF = {
    "clearsky_ghi": 27.15, "hour_cos": 20.83, "solar_elevation": 16.96,
    "shortwave_radiation": 12.30, "clearsky_index": 10.25, "effective_irradiance": 5.90,
    "hour_sin": 4.08, "wind_speed_10m": 0.48, "month_cos": 0.39,
    "temperature_2m": 0.38, "month_sin": 0.36, "relative_humidity_2m": 0.36,
    "cloud_cover": 0.31, "temp_loss_factor": 0.24,
}

IMPORTANCIA_XGB = {
    "clearsky_ghi": 39.56, "hour_cos": 28.56, "solar_elevation": 17.36,
    "hour_sin": 7.72, "shortwave_radiation": 1.96, "clearsky_index": 1.31,
    "month_sin": 0.76, "month_cos": 0.75, "effective_irradiance": 0.51,
    "wind_speed_10m": 0.33, "temperature_2m": 0.31, "temp_loss_factor": 0.29,
    "cloud_cover": 0.29, "relative_humidity_2m": 0.29,
}

# Pesos del ensamble
PESOS_ENSAMBLE = {
    "Random Forest": [1.0, 1.0, 0.0, 0.5, 0.5, 0.75],
    "XGBoost":       [0.0, 0.0, 1.0, 0.5, 0.5, 0.25],
}

SEGMENTOS = [0, 1, 2, 3, 4, 5]


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
        for i, (b, v) in enumerate(zip(bars, vals)):
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

    fig.suptitle("Métricas globales — Generación", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    ruta = CARPETA_FIGURAS / "fig_metricas_globales.png"
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    print(f"✓ {ruta}")


# ============================================================
# FIGURA 2 — Métricas diurnas
# ============================================================

def fig_metricas_diurnas():
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    modelos = ["Random Forest", "XGBoost"]
    colores = [COLOR_RF, COLOR_XGB]
    met = ["R2", "nRMSE", "nMAE", "MASE"]
    ylabels = ["R² diurno", "nRMSE (%)", "nMAE (%)", "MASE diurno"]

    for ax, m, yl in zip(axes, met, ylabels):
        vals = [METRICAS_DIURNAS[mod][m] for mod in modelos]
        bars = ax.bar(modelos, vals, color=colores, edgecolor="white", width=0.7)
        best = min(vals) if m != "R2" else max(vals)
        for b, v in zip(bars, vals):
            if v == best:
                b.set_edgecolor("#D62728")
                b.set_linewidth(2.5)
            ax.text(b.get_x() + b.get_width()/2, b.get_height()*1.02,
                    f"{v:.4f}" if m in ("R2", "MASE") else f"{v:.2f}",
                    ha="center", fontsize=9, fontweight="bold")
        ax.set_title(yl, fontweight="bold")
        ax.set_ylim(0, max(vals)*1.25)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticks(range(len(modelos)))
        ax.set_xticklabels(["RF", "XGB"])

    fig.suptitle("Métricas diurnas — Generación", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    ruta = CARPETA_FIGURAS / "fig_metricas_diurnas.png"
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    print(f"✓ {ruta}")


# ============================================================
# FIGURA 3 — MAE por segmento
# ============================================================

def fig_mae_segmento():
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(SEGMENTOS))
    w = 0.35
    rf = MAE_SEGMENTO["Random Forest"]
    xg = MAE_SEGMENTO["XGBoost"]

    ax.bar(x - w/2, rf, w, label="Random Forest", color=COLOR_RF, edgecolor="white")
    ax.bar(x + w/2, xg, w, label="XGBoost", color=COLOR_XGB, edgecolor="white")

    for i, (v1, v2) in enumerate(zip(rf, xg)):
        ax.text(i - w/2, v1 + 0.002, f"{v1:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, v2 + 0.002, f"{v2:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Seg {s}" for s in SEGMENTOS])
    ax.set_ylabel("MAE (kW)")
    ax.set_title("MAE por segmento — Generación", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    ruta = CARPETA_FIGURAS / "fig_mae_segmento.png"
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    print(f"✓ {ruta}")


# ============================================================
# FIGURA 4 — R² por segmento
# ============================================================

def fig_r2_segmento():
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(SEGMENTOS))
    w = 0.35
    rf = R2_SEGMENTO["Random Forest"]
    xg = R2_SEGMENTO["XGBoost"]

    ax.bar(x - w/2, rf, w, label="Random Forest", color=COLOR_RF, edgecolor="white")
    ax.bar(x + w/2, xg, w, label="XGBoost", color=COLOR_XGB, edgecolor="white")

    for i, (v1, v2) in enumerate(zip(rf, xg)):
        ax.text(i - w/2, v1 + 0.01, f"{v1:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, v2 + 0.01, f"{v2:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Seg {s}" for s in SEGMENTOS])
    ax.set_ylabel("R²")
    ax.set_title("R² por segmento — Generación", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    ruta = CARPETA_FIGURAS / "fig_r2_segmento.png"
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    print(f"✓ {ruta}")


# ============================================================
# FIGURA 5 — Importancia de características (RF)
# ============================================================

def fig_importancia_rf():
    fig, ax = plt.subplots(figsize=(10, 6))
    items = sorted(IMPORTANCIA_RF.items(), key=lambda x: x[1])
    nombres = [k for k, _ in items]
    valores = [v for _, v in items]

    bars = ax.barh(nombres, valores, color=COLOR_RF, edgecolor="white")
    for b, v in zip(bars, valores):
        ax.text(v + 0.3, b.get_y() + b.get_height()/2, f"{v:.2f}%", va="center", fontsize=8)

    ax.set_xlabel("Importancia (%)")
    ax.set_title("Importancia de características — Random Forest (Generación)", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    ruta = CARPETA_FIGURAS / "fig_importancia_rf.png"
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    print(f"✓ {ruta}")


# ============================================================
# FIGURA 6 — Importancia de características (XGB)
# ============================================================

def fig_importancia_xgb():
    fig, ax = plt.subplots(figsize=(10, 6))
    items = sorted(IMPORTANCIA_XGB.items(), key=lambda x: x[1])
    nombres = [k for k, _ in items]
    valores = [v for _, v in items]

    bars = ax.barh(nombres, valores, color=COLOR_XGB, edgecolor="white")
    for b, v in zip(bars, valores):
        ax.text(v + 0.3, b.get_y() + b.get_height()/2, f"{v:.2f}%", va="center", fontsize=8)

    ax.set_xlabel("Importancia (%)")
    ax.set_title("Importancia de características — XGBoost (Generación)", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    ruta = CARPETA_FIGURAS / "fig_importancia_xgb.png"
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    print(f"✓ {ruta}")


# ============================================================
# FIGURA 7 — Pesos del ensamble por segmento
# ============================================================

def fig_pesos_ensamble():
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(SEGMENTOS))
    rf = np.array(PESOS_ENSAMBLE["Random Forest"])
    xg = np.array(PESOS_ENSAMBLE["XGBoost"])

    ax.bar(x, rf, label="Random Forest", color=COLOR_RF, edgecolor="white")
    ax.bar(x, xg, bottom=rf, label="XGBoost", color=COLOR_XGB, edgecolor="white")

    for i, (v1, v2) in enumerate(zip(rf, xg)):
        ax.text(i, v1/2, f"{v1*100:.0f}%", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")
        ax.text(i, v1 + v2/2, f"{v2*100:.0f}%", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Seg {s}" for s in SEGMENTOS])
    ax.set_ylabel("Peso")
    ax.set_ylim(0, 1.05)
    ax.set_title("Pesos del ensamblaje por segmento — Generación", fontweight="bold")
    ax.legend(loc="upper right")
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
    print("  GENERANDO FIGURAS — GENERACIÓN")
    print("=" * 60)
    print(f"  Carpeta de salida: {CARPETA_FIGURAS}\n")

    fig_metricas_globales()
    fig_metricas_diurnas()
    fig_mae_segmento()
    fig_r2_segmento()
    fig_importancia_rf()
    fig_importancia_xgb()
    fig_pesos_ensamble()

    print("\n✓ Todas las figuras generadas correctamente")
    print("=" * 60)