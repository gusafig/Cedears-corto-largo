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
COMPLETO_CSV = "completo_tecnico.csv"
TOP_N = 20
PERIODO_HISTORIA = "1y"
MIN_RUEDAS_NECESARIAS = 210
TAMANO_LOTE = 25
PAUSA_ENTRE_LOTES = 3  # segundos

MODO_PRUEBA = os.environ.get("MODO_PRUEBA", "false").lower() == "true"
TICKERS_PRUEBA = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "MELI", "VALE", "UN", "PBR", "KO"]

VENTANA_DIVERGENCIA = 90  # ruedas hacia atrás para buscar divergencias recientes
UMBRAL_PIVOTE_PCT = 3.0   # % mínimo de movimiento para contar como un giro real (evita ruido)


def calcular_rsi(close, periodo=14):
    delta = close.diff()
    ganancia = delta.clip(lower=0)
    perdida = -delta.clip(upper=0)
    media_ganancia = ganancia.rolling(periodo).mean()
    media_perdida = perdida.rolling(periodo).mean()
    rs = media_ganancia / media_perdida
    return 100 - (100 / (1 + rs))


def encontrar_pivotes(serie, umbral_pct=UMBRAL_PIVOTE_PCT):
    """Zigzag simple: solo cuenta un giro si el precio se movió al menos
    `umbral_pct`% desde el último pivote. Evita que ruido chico del día a
    día se confunda con un máximo/mínimo real."""
    valores = serie.reset_index(drop=True).values
    if len(valores) < 3:
        return []
    pivotes = []
    tendencia = None
    ultimo_idx, ultimo_val = 0, valores[0]

    for i in range(1, len(valores)):
        if tendencia in (None, 'subiendo') and valores[i] >= ultimo_val:
            ultimo_val, ultimo_idx = valores[i], i
            if tendencia is None and (valores[i] - valores[0]) / valores[0] * 100 >= umbral_pct:
                tendencia = 'subiendo'
        elif tendencia in (None, 'bajando') and valores[i] <= ultimo_val:
            ultimo_val, ultimo_idx = valores[i], i
            if tendencia is None and (valores[i] - valores[0]) / valores[0] * 100 <= -umbral_pct:
                tendencia = 'bajando'
        elif tendencia == 'subiendo':
            if (valores[i] - ultimo_val) / ultimo_val * 100 <= -umbral_pct:
                pivotes.append((ultimo_idx, ultimo_val, 'max'))
                tendencia, ultimo_val, ultimo_idx = 'bajando', valores[i], i
        elif tendencia == 'bajando':
            if (valores[i] - ultimo_val) / ultimo_val * 100 >= umbral_pct:
                pivotes.append((ultimo_idx, ultimo_val, 'min'))
                tendencia, ultimo_val, ultimo_idx = 'subiendo', valores[i], i

    pivotes.append((ultimo_idx, ultimo_val, 'max' if tendencia == 'subiendo' else 'min'))
    return pivotes


def detectar_divergencia(close, oscilador, ventana=VENTANA_DIVERGENCIA):
    """Compara los dos últimos mínimos (o máximos) del precio contra el
    oscilador (RSI o línea MACD) para detectar divergencia alcista o bajista."""
    close_r = close.iloc[-ventana:].reset_index(drop=True)
    osc_r = oscilador.iloc[-ventana:].reset_index(drop=True)
    if osc_r.isna().all():
        return "ninguna"

    pivotes = encontrar_pivotes(close_r)
    minimos = [p for p in pivotes if p[2] == 'min']
    maximos = [p for p in pivotes if p[2] == 'max']

    if len(minimos) >= 2:
        (i1, v1, _), (i2, v2, _) = minimos[-2], minimos[-1]
        if v2 < v1 and not pd.isna(osc_r.iloc[i1]) and not pd.isna(osc_r.iloc[i2]) and osc_r.iloc[i2] > osc_r.iloc[i1]:
            return "alcista"
    if len(maximos) >= 2:
        (i1, v1, _), (i2, v2, _) = maximos[-2], maximos[-1]
        if v2 > v1 and not pd.isna(osc_r.iloc[i1]) and not pd.isna(osc_r.iloc[i2]) and osc_r.iloc[i2] < osc_r.iloc[i1]:
            return "bajista"
    return "ninguna"


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


VENTANA_CRUCE = 15  # ruedas hacia atrás para considerar un cruce "reciente"


def detectar_cruce_reciente(sma50, sma200, ventana=VENTANA_CRUCE):
    """Devuelve si SMA50 y SMA200 se cruzaron en las últimas `ventana` ruedas,
    y de qué tipo: 'dorado' (SMA50 pasa por ENCIMA, señal alcista) o
    'de la muerte' (SMA50 pasa por DEBAJO, señal bajista)."""
    diferencia = (sma50 - sma200).iloc[-(ventana + 1):]
    cruce_reciente, tipo_cruce = False, "ninguno"
    for i in range(1, len(diferencia)):
        anterior, actual = diferencia.iloc[i - 1], diferencia.iloc[i]
        if pd.isna(anterior) or pd.isna(actual):
            continue
        if anterior <= 0 and actual > 0:
            cruce_reciente, tipo_cruce = True, "dorado"
        elif anterior >= 0 and actual < 0:
            cruce_reciente, tipo_cruce = True, "de la muerte"
    return cruce_reciente, tipo_cruce


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
    cruce_reciente, tipo_cruce = detectar_cruce_reciente(sma50, sma200)

    roc = (precio_actual - float(close.iloc[-21])) / float(close.iloc[-21]) * 100

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    macd_hist = float((macd_line - signal_line).iloc[-1])
    # Normalizado como % del precio: así una acción de $50.000 y una de $50
    # con el mismo momentum relativo dan un valor comparable.
    macd_hist_pct = macd_hist / precio_actual * 100

    vol_promedio_20 = float(volume.rolling(20).mean().iloc[-1])
    vol_relativo = float(volume.iloc[-1]) / vol_promedio_20 if vol_promedio_20 > 0 else np.nan

    rsi = calcular_rsi(close)
    divergencia_rsi = detectar_divergencia(close, rsi)
    divergencia_macd = detectar_divergencia(close, macd_line)

    # A/D Line normalizada: promedio de CLV ponderado por volumen en los
    # últimos 20 ruedas. Queda acotado entre -1 y 1, comparable entre
    # empresas sin importar su escala de volumen.
    rango = (high - low).replace(0, np.nan)
    clv = ((close - low) - (high - close)) / rango
    clv_reciente = clv.iloc[-20:]
    volumen_reciente = volume.iloc[-20:]
    volumen_total = float(volumen_reciente.sum())
    ad_normalizado = (
        float((clv_reciente * volumen_reciente).fillna(0).sum() / volumen_total)
        if volumen_total > 0 else np.nan
    )

    return {
        "tendencia": tendencia,
        "cruce_reciente": cruce_reciente,
        "tipo_cruce": tipo_cruce,
        "distancia_media_pct": distancia_media,
        "roc_20d_pct": roc,
        "macd_hist_pct": macd_hist_pct,
        "volumen_relativo": vol_relativo,
        "ad_normalizado": ad_normalizado,
        "divergencia_rsi": divergencia_rsi,
        "divergencia_macd": divergencia_macd,
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
    tabla["pct_macd"] = percentil(tabla["macd_hist_pct"])
    tabla["pct_volumen"] = percentil(tabla["volumen_relativo"])
    tabla["pct_ad"] = percentil(tabla["ad_normalizado"])
    tabla["pct_tendencia"] = tabla["tendencia"] * 100

    tabla["puntaje_tecnico"] = tabla[
        ["pct_tendencia", "pct_distancia", "pct_roc", "pct_macd", "pct_volumen", "pct_ad"]
    ].mean(axis=1)

    tabla = tabla.sort_values("puntaje_tecnico", ascending=False)

    columnas_salida = [
        "ticker_byma", "nombre_empresa", "puntaje_tecnico", "tendencia",
        "cruce_reciente", "tipo_cruce", "divergencia_rsi", "divergencia_macd",
        "distancia_media_pct", "roc_20d_pct", "macd_hist_pct", "volumen_relativo", "ad_normalizado",
    ]
    tabla_completa = tabla[columnas_salida].round(2)
    tabla_completa.to_csv(COMPLETO_CSV, index=False)  # todo el universo, para el buscador
    tabla_completa.head(TOP_N if not MODO_PRUEBA else len(tabla)).to_csv(SALIDA_CSV, index=False)  # solo el Top N

    print(f"\nProcesados con éxito: {len(tabla)}/{len(tickers)} tickers. Fallidos: {len(fallidos)}.")
    print(f"Top {TOP_N} guardado en {SALIDA_CSV}, universo completo en {COMPLETO_CSV}")
    if fallidos:
        print("Tickers sin datos suficientes (no rompen el proceso):")
        print(", ".join(fallidos))


if __name__ == "__main__":
    main()
