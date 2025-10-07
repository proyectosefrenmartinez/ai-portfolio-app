import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from io import StringIO

# --- Config ---
st.set_page_config(page_title="Comparador de Cartera vs Benchmarks", layout="wide")
st.title("Comparador de Cartera vs Benchmarks")
st.caption("Sube un CSV de pesos por fecha y compara tu cartera frente a S&P500, MSCI World, Bonos, Monetarios y un proxy ROBO 60/40.")

# --- Sidebar params ---
st.sidebar.header("Parámetros")
start_date = st.sidebar.date_input("Fecha inicial (para descargar benchmarks)", pd.to_datetime("2010-01-01"))
cost_bps = st.sidebar.number_input("Coste por rebalanceo (bps por lado)", 0, 50, 10)
st.sidebar.caption("Consejo: para empezar, deja 10 bps.")

# --- Benchmarks (puedes cambiarlos más adelante) ---
BENCH_TICKERS = {
    "S&P 500 (SPY)": "SPY",            # USA acciones
    "MSCI World (IWDA.AS)": "IWDA.AS", # World UCITS (EUR/AMS) - si falla, puedes probar 'URTH'
    "Bonos USA (AGG)": "AGG",          # Bonos agregados
    "Monetarios (BIL)": "BIL"          # T-Bills 1-3m
}

# --- Utils ---
@st.cache_data(show_spinner=False)
def download_prices(tickers, start):
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Adj Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    # índice a diario -> quedarnos con meses
    data_m = data.resample("M").last().dropna(how="all")
    return data_m

def compute_turnover_cost(weights_m, bps):
    turnover = weights_m.diff().abs().sum(axis=1).fillna(0.0)
    return turnover * (bps/10000.0)

def portfolio_from_weights(weights_df, prices_df, cost_bps=10):
    prices_m = prices_df.copy()
    # Alinear pesos a mes y mismas fechas
    weights_df["date"] = pd.to_datetime(weights_df["date"])
    weights_m = weights_df.set_index("date").resample("M").last().ffill()
    common_cols = [c for c in weights_m.columns if c in prices_m.columns]
    if not common_cols:
        raise ValueError("Tus columnas de pesos no coinciden con ningún ticker descargado.")
    prices_m = prices_m[common_cols].dropna(how="any")
    weights_m = weights_m[common_cols].loc[prices_m.index]
    # Normalizar por seguridad
    weights_m = weights_m.div(weights_m.abs().sum(axis=1), axis=0).fillna(0.0)

    rets = prices_m.pct_change().fillna(0.0)
    tcost = compute_turnover_cost(weights_m, cost_bps)
    port_ret = (weights_m.shift(1).fillna(0.0) * rets).sum(axis=1) - tcost
    equity = (1 + port_ret).cumprod()
    return port_ret, equity, weights_m

def metrics_from_returns(r):
    r = r.dropna()
    if len(r) == 0:
        return np.nan, np.nan, np.nan, np.nan
    n = len(r)
    total = (1+r).prod() - 1
    cagr = (1+total)**(12/n) - 1
    vol_a = r.std() * np.sqrt(12)
    sharpe = cagr/vol_a if vol_a > 0 else np.nan
    eq = (1+r).cumprod()
    dd = (eq/eq.cummax() - 1).min()
    return cagr, vol_a, sharpe, dd

# --- Descarga benchmarks ---
bench_px = download_prices(list(BENCH_TICKERS.values()), start_date)
bench_ret = bench_px.pct_change().dropna()

# Proxy Robo 60/40 (IWDA + AGG)
if set(["IWDA.AS","AGG"]).issubset(bench_ret.columns):
    robo_ret = 0.6*bench_ret["IWDA.AS"] + 0.4*bench_ret["AGG"]
    bench_ret["ROBO_60_40"] = robo_ret

# --- Ejemplo de CSV (descargable) ---
example_csv = """date,SPY,IWDA.AS,AGG,BIL,CASH
2018-01-31,0.60,0.20,0.10,0.05,0.05
2020-01-31,0.55,0.25,0.10,0.05,0.05
2022-01-31,0.50,0.30,0.10,0.05,0.05
2024-01-31,0.45,0.35,0.10,0.05,0.05
"""

st.download_button(
    "⬇️ Descargar CSV de ejemplo (pesos por fecha)",
    data=example_csv,
    file_name="weights_example.csv",
    mime="text/csv"
)

uploaded = st.file_uploader("Sube tu archivo de pesos (formato ancho: columnas = date, TICKERS... que sumen ≈ 1 por fila)", type=["csv"])

# --- Procesado principal ---
if uploaded:
    weights = pd.read_csv(uploaded)
    # Descargar también precios de los tickers del usuario que NO sean CASH/EUR/USD
    user_tickers = [c for c in weights.columns if c not in ["date","CASH","EUR","USD"]]
    user_px = download_prices(user_tickers, pd.to_datetime(weights["date"]).min())

    # Si el usuario incluye CASH, creamos serie plana = 1 (rentabilidad ~0)
    if "CASH" in weights.columns:
        user_px["CASH"] = 1.0

    # Juntamos con benchmarks para poder comparar en un mismo gráfico
    all_px = user_px.join(bench_px, how="outer")

    # Cartera del usuario
    port_ret, port_eq, _ = portfolio_from_weights(weights, all_px, cost_bps=cost_bps)

    # Gráfico
    st.subheader("Curva de capital (base = 1)")
    chart_df = pd.DataFrame({"Tu cartera": port_eq})
    bench_eq = (1+bench_ret).cumprod()
    # Renombrar columnas para legibilidad
    nice_names = {v:k for k,v in BENCH_TICKERS.items()}
    bench_eq = bench_eq.rename(columns=nice_names)
    if "ROBO_60_40" in bench_eq.columns:
        pass
    else:
        # Si hemos renombrado columnas, añadimos ROBO si existe en bench_ret original
        if "ROBO_60_40" in bench_ret.columns:
            bench_eq["ROBO_60_40"] = (1+bench_ret["ROBO_60_40"]).cumprod()

    chart = chart_df.join(bench_eq, how="inner")
    st.line_chart(chart)

    # Métricas
    st.subheader("Métricas")
    rows = []
    rows.append(("Tu cartera", *metrics_from_returns(port_ret)))
    # benchmarks
    # Para métricas usamos bench_ret original (sin renombrar) y mostramos nombre bonito si aplica
    for col in bench_ret.columns:
        m = metrics_from_returns(bench_ret[col])
        display_name = nice_names.get(col, col)
        rows.append((display_name, *m))
    metrics_df = pd.DataFrame(rows, columns=["Estrategia","CAGR","Volatilidad","Sharpe","Max Drawdown"])
    st.dataframe(metrics_df.style.format({
        "CAGR":"{:.2%}","Volatilidad":"{:.2%}","Sharpe":"{:.2f}","Max Drawdown":"{:.2%}"
    }))

    st.caption("Nota: ROBO_60_40 es un proxy del ‘promedio de roboadvisors’ (= 60% MSCI World + 40% Bonos).")
else:
    st.info("💡 Descarga el CSV de ejemplo, ajústalo con tus pesos y súbelo para ver tu curva y comparativas.")
