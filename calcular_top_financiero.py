"""
Calcula el puntaje financiero para el universo de CEDEARs (cedears_byma.csv)
y genera un ranking con los Top N mejores: top_financiero.csv

Indicadores usados (todos con el mismo peso):
  - Valuación: P/E y EV/EBITDA (más bajo = mejor)
  - Rentabilidad: ROE (más alto = mejor)
  - Solvencia: Deuda / EBITDA (más bajo = mejor)
  - Crecimiento: variación de ingresos interanual (más alto = mejor)

Los fundamentales corresponden a la EMPRESA SUBYACENTE (la que cotiza en su
bolsa de origen, ej. NYSE/NASDAQ), no al CEDEAR en sí — por eso se pide el
ticker "de origen", no el de BYMA. En la gran mayoría de los casos son el
mismo código, pero hay excepciones conocidas (ver TICKERS_EXCEPCION).

Al igual que en el script técnico:
  - se usa una sesión que simula un navegador para evitar bloqueos
  - hay un MODO_PRUEBA para validar rápido con pocos tickers antes de
    correr sobre el universo completo
  - se corre semanalmente (no hace falta actualizar balances todos los días)
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
    SESSION = None

UNIVERSO_CSV = "cedears_byma.csv"
SALIDA_CSV = "top_financiero.csv"
COMPLETO_CSV = "completo_financiero.csv"
TOP_N = 20
PAUSA_ENTRE_TICKERS = 1.5  # segundos

MODO_PRUEBA = os.environ.get("MODO_PRUEBA", "false").lower() == "true"
TICKERS_PRUEBA = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "MELI", "VALE", "UN", "PBR", "KO"]

# Casos donde el código de BYMA no coincide con el ticker real en la bolsa
# de origen. Sumar acá cualquier otro caso que se detecte más adelante.
TICKERS_EXCEPCION = {
    "UN": "NU",     # NU Holdings (Nubank)
    "DISN": "DIS",  # The Walt Disney Company
    "XROX": "XRX",  # Xerox Holding Corporation
}

# Sufijo que hay que agregarle al ticker según la bolsa de origen, para que
# Yahoo Finance lo reconozca (EEUU no necesita sufijo).
SUFIJOS_POR_MERCADO = {
    "B3": ".SA",
    "BOVESPA": ".SA",
    "FRANKFURT": ".DE",
    "XETRA": ".DE",
    "LONDON STOCK EXCHANGE": ".L",
}


def ticker_de_origen(ticker_byma, mercado=""):
    base = TICKERS_EXCEPCION.get(ticker_byma, ticker_byma)
    base = base.replace(".", "-")  # convención de Yahoo para clases de acciones (ej. BRK-B)
    sufijo = SUFIJOS_POR_MERCADO.get(str(mercado).strip(), "")
    return f"{base}{sufijo}"


def a_numero(valor):
    """Convierte a float; si no se puede (texto raro, None, etc.), devuelve NaN
    en vez de romper el cálculo más adelante."""
    try:
        return float(valor)
    except (TypeError, ValueError):
        return np.nan


def obtener_fundamentales(ticker_byma, mercado=""):
    """Descarga los datos fundamentales de la empresa subyacente para un ticker."""
    simbolo = ticker_de_origen(ticker_byma, mercado)
    try:
        info = yf.Ticker(simbolo, session=SESSION).info if SESSION else yf.Ticker(simbolo).info
    except Exception:
        return None

    if not info or (info.get("trailingPE") is None and info.get("enterpriseToEbitda") is None):
        return None

    ebitda = a_numero(info.get("ebitda"))
    deuda_total = a_numero(info.get("totalDebt"))
    deuda_ebitda = (deuda_total / ebitda) if (ebitda and deuda_total and ebitda > 0) else np.nan

    return {
        "pe": a_numero(info.get("trailingPE")),
        "ev_ebitda": a_numero(info.get("enterpriseToEbitda")),
        "roe": a_numero(info.get("returnOnEquity")),
        "deuda_ebitda": a_numero(deuda_ebitda),
        "crecimiento_ingresos": a_numero(info.get("revenueGrowth")),
    }


def percentil(serie, invertir=False):
    """Percentil 0-100. Si invertir=True, un valor más BAJO obtiene percentil más ALTO
    (para métricas donde 'más barato'/'menos deuda' es mejor)."""
    return serie.rank(pct=True, ascending=not invertir) * 100


def main():
    universo = pd.read_csv(UNIVERSO_CSV)

    if MODO_PRUEBA:
        universo = universo[universo["ticker_byma"].isin(TICKERS_PRUEBA)]
        print(f"MODO PRUEBA activado: usando {len(universo)} tickers.")

    resultados = []
    fallidos = []

    for _, fila in universo.iterrows():
        ticker = str(fila["ticker_byma"]).strip()
        mercado = fila.get("mercado", "")
        datos = obtener_fundamentales(ticker, mercado)
        if datos is None:
            fallidos.append(ticker)
        else:
            datos["ticker_byma"] = ticker
            datos["nombre_empresa"] = fila["nombre_empresa"]
            resultados.append(datos)
        time.sleep(PAUSA_ENTRE_TICKERS)

    if not resultados:
        print("No se pudo calcular ningún indicador financiero. Revisar conexión o datos de origen.")
        sys.exit(1)

    tabla = pd.DataFrame(resultados)
    for columna in ["pe", "ev_ebitda", "roe", "deuda_ebitda", "crecimiento_ingresos"]:
        tabla[columna] = pd.to_numeric(tabla[columna], errors="coerce")

    tabla["pct_pe"] = percentil(tabla["pe"], invertir=True)
    tabla["pct_ev_ebitda"] = percentil(tabla["ev_ebitda"], invertir=True)
    tabla["pct_roe"] = percentil(tabla["roe"], invertir=False)
    tabla["pct_deuda_ebitda"] = percentil(tabla["deuda_ebitda"], invertir=True)
    tabla["pct_crecimiento"] = percentil(tabla["crecimiento_ingresos"], invertir=False)

    columnas_pct = ["pct_pe", "pct_ev_ebitda", "pct_roe", "pct_deuda_ebitda", "pct_crecimiento"]
    tabla["n_metricas"] = tabla[columnas_pct].notna().sum(axis=1)
    tabla["puntaje_financiero"] = tabla[columnas_pct].mean(axis=1, skipna=True)

    MIN_METRICAS = 3
    excluidos_por_datos = tabla[tabla["n_metricas"] < MIN_METRICAS]["ticker_byma"].tolist()
    tabla = tabla[tabla["n_metricas"] >= MIN_METRICAS]

    tabla = tabla.sort_values("puntaje_financiero", ascending=False)

    columnas_salida = [
        "ticker_byma", "nombre_empresa", "puntaje_financiero", "n_metricas",
        "pe", "ev_ebitda", "roe", "deuda_ebitda", "crecimiento_ingresos",
    ]
    tabla_completa = tabla[columnas_salida].round(2)
    tabla_completa.to_csv(COMPLETO_CSV, index=False)  # todo el universo, para el buscador
    tabla_completa.head(TOP_N if not MODO_PRUEBA else len(tabla)).to_csv(SALIDA_CSV, index=False)

    print(f"\nProcesados con éxito: {len(resultados)}/{len(universo)} tickers. Fallidos: {len(fallidos)}.")
    if excluidos_por_datos:
        print(f"Excluidos del ranking por tener menos de {MIN_METRICAS} métricas disponibles: {', '.join(excluidos_por_datos)}")
    print(f"Top {TOP_N} guardado en {SALIDA_CSV}, universo completo en {COMPLETO_CSV}")
    if fallidos:
        print("Tickers sin datos fundamentales disponibles (no rompen el proceso):")
        print(", ".join(fallidos))


if __name__ == "__main__":
    main()
