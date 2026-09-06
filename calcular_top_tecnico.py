"""
Calcula el puntaje técnico para el universo de CEDEARs (cedears_byma.csv)
y genera un ranking con los Top N mejores: top_tecnico.csv

Indicadores usados (todos con el mismo peso):
  - Tendencia (SMA50 vs SMA200)
  - Distancia a la media de 200 ruedas
  - Momentum: ROC de 20 ruedas y MACD (histograma)
  - Volumen relativo (vs promedio de 20 ruedas)
  - A/D Line (acumulación/distribución), pendiente de las últimas 20 ruedas

Pensado para correr dentro de GitHub Actions. Para evitar que Yahoo Finance
bloquee los pedidos (algo común cuando vienen muchos seguidos desde
servidores compartidos como los de GitHub), este script:
  - pide los precios de a LOTES en vez de ticker por ticker
  - simula un navegador real en vez de un script
  - espera un poco entre lotes

Variable de entorno MODO_PRUEBA=true: corre solo sobre un puñado de
tickers conocidos, para validar rápido que todo funciona antes de
lanzar la corrida completa sobre los 413.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import yfinance as yf

try:
    from curl_cffi import requests as cffi_requests
    SESSION = cffi_requests.Session(impersonate="chrome")
except Exception:
    SESSION = None  # si no está disponible, seguimos sin sesión especial

UNIVERSO_CSV = "cedears_byma.csv"
SALIDA_CSV = "top_tecnico.csv"
TOP_N = 20
PERIODO_HISTORIA = "1y"
MIN_RUEDAS_NECESARIAS = 210
TAMANO_LOTE = 25
PAUSA_ENTRE_LOTES = 3  # segundos

MODO_PRUEBA = os.environ.get("MODO_PRUEBA", "false").lower() == "true"
TICKERS_PRUEBA = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "MELI", "VALE", "UN", "PBR", "KO"]


def descargar_lote(tickers_base, sufijo):
    """Descarga un lote de tickers en una sola llamada (menos pedidos = menos riesgo de bloqueo)."""
    simbolos = [f"{t}{sufijo}" for t in tickers_base]
    kwargs = dict(period=PERIODO_HISTORIA, progress=False, auto_adjust=True,
                  group_by="ticker", threads=True)
    if SESSION is not None:
        kwargs["session"] = SESSION
    try:
        data = yf.download(simbolos, **kwargs)
    except Exception as e:
        print(f"  Lote falló ({sufijo}): {e}")
        return {}

    resultado = {}
    for base, simbolo in zip(tickers_base, simbolos):
        try:
            df = data[simbolo] if len(simbolos) > 1 else data
            if df is not None and not df.dropna(how="all").empty and len(df) >= MIN_RUEDAS_NECESARIAS:
                resultado[base] = df
        except Exception:
            continue
    return resultado


def obtener_precios_universo(tickers):
    """
    Recorre todos los tickers en lotes, primero probando el sufijo ".BA",
    y reintentando solo los que fallaron con la variante "D.BA"
    (algunos CEDEARs cambian de serie y necesitan ese sufijo).
    """
    precios = {}
    pendientes = list(tickers)

    for sufijo in [".BA", "D.BA"]:
        if not pendientes:
            break
        print(f"Probando sufijo '{sufijo}' para {len(pendientes)} tickers...")
        nuevos_pendientes = []
        for i in range(0, len(pendientes), TAMANO_LOTE):
            lote = pendientes[i:i + TAMANO_LOTE]
            encontrados = descargar_lote(lote, sufijo)
            precios.update(encontrados)
            faltantes = [t for t in lote if t not in encontrados]
            nuevos_pendientes.extend(faltantes)
            print(f"  Lote {i // TAMANO_LOTE + 1}: {len(encontrados)}/{len(lote)} OK")
            time.sleep(PAUSA_ENTRE_LOTES)
        pendientes = nuevos_pendientes

    return precios, pendientes  # pendientes = los que fallaron con ambos sufijos


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

    if MODO_PRUEBA:
        universo = universo[universo["ticker_byma"].isin(TICKERS_PRUEBA)]
        print(f"MODO PRUEBA activado: usando {len(universo)} tickers de {len(TICKERS_PRUEBA)} esperados.")

    nombres_por_ticker = dict(zip(universo["ticker_byma"], universo["nombre_empresa"]))
    tickers = list(nombres_por_ticker.keys())

    precios, fallidos = obtener_precios_universo(tickers)

    resultados = []
    for ticker, df in precios.items():
        try:
            ind = calcular_indicadores(df)
            ind["ticker_byma"] = ticker
            ind["nombre_empresa"] = nombres_por_ticker[ticker]
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
    top = tabla[columnas_salida].head(TOP_N if not MODO_PRUEBA else len(tabla)).round(2)
    top.to_csv(SALIDA_CSV, index=False)

    print(f"\nProcesados con éxito: {len(tabla)}/{len(tickers)} tickers. Fallidos: {len(fallidos)}.")
    print(f"Resultado guardado en {SALIDA_CSV}")
    if fallidos:
        print("Tickers sin datos suficientes (no rompen el proceso):")
        print(", ".join(fallidos))


if __name__ == "__main__":
    main()
