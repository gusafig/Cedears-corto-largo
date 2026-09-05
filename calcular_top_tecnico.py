"""
Calcula el puntaje técnico para el universo de CEDEARs (cedears_byma.csv)
y genera un ranking con los Top N mejores: top_tecnico.csv

Indicadores usados (todos con el mismo peso):
  - Tendencia (SMA50 vs SMA200)
  - Distancia a la media de 200 ruedas
  - Momentum: ROC de 20 ruedas y MACD (histograma)
  - Volumen relativo (vs promedio de 20 ruedas)
  - A/D Line (acumulación/distribución), pendiente de las últimas 20 ruedas

Este script está pensado para correr dentro de GitHub Actions, donde sí hay
acceso a internet para descargar precios. No requiere que el usuario sepa
programar: una vez subido al repositorio, el workflow lo ejecuta solo.
"""

import sys
import pandas as pd
import numpy as np
import yfinance as yf

UNIVERSO_CSV = "cedears_byma.csv"
SALIDA_CSV = "top_tecnico.csv"
TOP_N = 20
PERIODO_HISTORIA = "1y"
MIN_RUEDAS_NECESARIAS = 210  # para poder calcular SMA200 con margen


def obtener_precios(ticker_byma):
    """
    Intenta descargar precios para un CEDEAR de BYMA.
    Prueba variantes de sufijo porque algunos tickers de BYMA cambian de
    serie periódicamente (ej: re-emisiones) y el sufijo ".BA" simple no
    siempre alcanza.
    """
    variantes = [f"{ticker_byma}.BA", f"{ticker_byma}D.BA"]
    for variante in variantes:
        try:
            df = yf.download(variante, period=PERIODO_HISTORIA, progress=False, auto_adjust=True)
            if df is not None and len(df) >= MIN_RUEDAS_NECESARIAS:
                return df
        except Exception:
            continue
    return None


def calcular_indicadores(df):
    close = df["Close"]
    volume = df["Volume"]
    high = df["High"]
    low = df["Low"]

    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    precio_actual = float(close.iloc[-1])
    sma50_actual = float(sma50.iloc[-1])
    sma200_actual = float(sma200.iloc[-1])

    tendencia = 1 if sma50_actual > sma200_actual else 0
    distancia_media = (precio_actual - sma200_actual) / sma200_actual * 100

    roc = (precio_actual - float(close.iloc[-21])) / float(close.iloc[-21]) * 100

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    macd_hist = float((macd_line - signal_line).iloc[-1])

    vol_promedio_20 = float(volume.rolling(20).mean().iloc[-1])
    vol_relativo = float(volume.iloc[-1]) / vol_promedio_20 if vol_promedio_20 > 0 else np.nan

    # A/D Line: pondera el cierre dentro del rango del día por el volumen
    rango = (high - low).replace(0, np.nan)
    clv = ((close - low) - (high - close)) / rango
    ad = (clv * volume).fillna(0).cumsum()
    ad_pendiente = float(ad.iloc[-1] - ad.iloc[-20])

    return {
        "tendencia": tendencia,
        "distancia_media_pct": distancia_media,
        "roc_20d_pct": roc,
        "macd_hist": macd_hist,
        "volumen_relativo": vol_relativo,
        "ad_pendiente": ad_pendiente,
    }


def percentil(serie):
    return serie.rank(pct=True) * 100


def main():
    universo = pd.read_csv(UNIVERSO_CSV)
    resultados = []
    fallidos = []

    for _, fila in universo.iterrows():
        ticker = str(fila["ticker_byma"]).strip()
        df = obtener_precios(ticker)
        if df is None:
            fallidos.append(ticker)
            continue
        try:
            ind = calcular_indicadores(df)
            ind["ticker_byma"] = ticker
            ind["nombre_empresa"] = fila["nombre_empresa"]
            resultados.append(ind)
        except Exception:
            fallidos.append(ticker)

    if not resultados:
        print("No se pudo calcular ningún indicador. Revisar conexión o datos de origen.")
        sys.exit(1)

    tabla = pd.DataFrame(resultados)

    tabla["pct_distancia"] = percentil(tabla["distancia_media_pct"])
    tabla["pct_roc"] = percentil(tabla["roc_20d_pct"])
    tabla["pct_macd"] = percentil(tabla["macd_hist"])
    tabla["pct_volumen"] = percentil(tabla["volumen_relativo"])
    tabla["pct_ad"] = percentil(tabla["ad_pendiente"])
    tabla["pct_tendencia"] = tabla["tendencia"] * 100

    tabla["puntaje_tecnico"] = tabla[
        ["pct_tendencia", "pct_distancia", "pct_roc", "pct_macd", "pct_volumen", "pct_ad"]
    ].mean(axis=1)

    tabla = tabla.sort_values("puntaje_tecnico", ascending=False)

    columnas_salida = [
        "ticker_byma", "nombre_empresa", "puntaje_tecnico", "tendencia",
        "distancia_media_pct", "roc_20d_pct", "macd_hist", "volumen_relativo", "ad_pendiente",
    ]
    tabla[columnas_salida].head(TOP_N).round(2).to_csv(SALIDA_CSV, index=False)

    print(f"Procesados: {len(tabla)} tickers con datos. Fallidos: {len(fallidos)}.")
    print(f"Top {TOP_N} guardado en {SALIDA_CSV}")
    if fallidos:
        print("Tickers sin datos suficientes (se ignoran, no rompen el proceso):")
        print(", ".join(fallidos))


if __name__ == "__main__":
    main()
