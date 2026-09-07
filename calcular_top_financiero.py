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
TOP_N = 20
PAUSA_ENTRE_TICKERS = 1.5  # segundos

MODO_PRUEBA = os.environ.get("MODO_PRUEBA", "false").lower() == "true"
TICKERS_PRUEBA = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "MELI", "VALE", "UN", "PBR", "KO"]

# Casos donde el código de BYMA no coincide con el ticker real en la bolsa
# de origen. Sumar acá cualquier otro caso que se detecte más adelante.
TICKERS_EXCEPCION = {
    "UN": "NU",  # NU Holdings (Nubank): BYMA lo codifica como UN, el ticker real es NU
}


def ticker_de_origen(ticker_byma):
    return TICKERS_EXCEPCION.get(ticker_byma, ticker_byma)


def obtener_fundamentales(ticker_byma):
    """Descarga los datos fundamentales de la empresa subyacente para un ticker."""
    simbolo = ticker_de_origen(ticker_byma)
    try:
        info = yf.Ticker(simbolo, session=SESSION).info if SESSION else yf.Ticker(simbolo).info
    except Exception:
        return None

    if not info or info.get("trailingPE") is None and info.get("enterpriseToEbitda") is None:
        return None

    ebitda = info.get("ebitda")
    deuda_total = info.get("totalDebt")
    deuda_ebitda = (deuda_total / ebitda) if (ebitda and deuda_total and ebitda > 0) else np.nan

    return {
        "pe": info.get("trailingPE", np.nan),
        "ev_ebitda": info.get("enterpriseToEbitda", np.nan),
        "roe": info.get("returnOnEquity", np.nan),
        "deuda_ebitda": deuda_ebitda,
        "crecimiento_ingresos": info.get("revenueGrowth", np.nan),
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
        datos = obtener_fundamentales(ticker)
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

    tabla["pct_pe"] = percentil(tabla["pe"], invertir=True)
    tabla["pct_ev_ebitda"] = percentil(tabla["ev_ebitda"], invertir=True)
    tabla["pct_roe"] = percentil(tabla["roe"], invertir=False)
    tabla["pct_deuda_ebitda"] = percentil(tabla["deuda_ebitda"], invertir=True)
    tabla["pct_crecimiento"] = percentil(tabla["crecimiento_ingresos"], invertir=False)

    tabla["puntaje_financiero"] = tabla[
        ["pct_pe", "pct_ev_ebitda", "pct_roe", "pct_deuda_ebitda", "pct_crecimiento"]
    ].mean(axis=1, skipna=True)

    tabla = tabla.sort_values("puntaje_financiero", ascending=False)

    columnas_salida = [
        "ticker_byma", "nombre_empresa", "puntaje_financiero",
        "pe", "ev_ebitda", "roe", "deuda_ebitda", "crecimiento_ingresos",
    ]
    top = tabla[columnas_salida].head(TOP_N if not MODO_PRUEBA else len(tabla)).round(2)
    top.to_csv(SALIDA_CSV, index=False)

    print(f"\nProcesados con éxito: {len(tabla)}/{len(universo)} tickers. Fallidos: {len(fallidos)}.")
    print(f"Resultado guardado en {SALIDA_CSV}")
    if fallidos:
        print("Tickers sin datos fundamentales disponibles (no rompen el proceso):")
        print(", ".join(fallidos))


if __name__ == "__main__":
    main()
