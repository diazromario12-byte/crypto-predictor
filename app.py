import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Crypto Price Predictor",
    page_icon="📈",
    layout="wide"
)

CRYPTO_LIST = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD",
    "DOGE-USD", "AVAX-USD", "DOT-USD", "LINK-USD", "MATIC-USD",
    "UNI-USD", "SHIB-USD", "LTC-USD", "BCH-USD", "ATOM-USD",
    "XLM-USD", "TRX-USD", "FIL-USD", "APT-USD", "ARB-USD",
    "SUI-USD", "OP-USD", "INJ-USD", "PEPE-USD", "FLOKI-USD",
    "AAVE-USD", "ALGO-USD", "NEAR-USD", "ICP-USD", "SAND-USD",
    "MANA-USD", "AXS-USD", "THETA-USD", "FTM-USD", "GRT-USD",
    "RUNE-USD", "EGLD-USD", "EOS-USD", "KAVA-USD", "ENS-USD",
    "CRV-USD", "BAT-USD", "ZEC-USD", "DASH-USD", "XMR-USD",
]

@st.cache_data(ttl=3600)
def fetch_data(ticker, period="2y"):
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        return df
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df

def prepare_series(df):
    series = df[["Close"]].reset_index().rename(
        columns={"Date": "ds", "Close": "y"}
    )
    series["unique_id"] = "crypto"
    return series

def train_arima(series, forecast_days):
    model = AutoARIMA(season_length=7)
    model.fit(y=series["y"].values.flatten())
    pred = model.predict(h=forecast_days)
    forecast = pd.DataFrame({"yhat": pred["mean"]})
    fitted = series[["ds", "y"]].copy()
    fitted["yhat"] = model.predict_in_sample()["fitted"]
    return fitted, forecast

def train_prophet(series, forecast_days):
    df = series[["ds", "y"]].copy()
    df.columns = ["ds", "y"]
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    model.fit(df)
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    return model, forecast

def compute_metrics(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    valid = ~(np.isnan(actual) | np.isnan(predicted))
    actual, predicted = actual[valid], predicted[valid]
    if len(actual) == 0:
        return 0, 0
    return mean_absolute_error(actual, predicted), np.sqrt(mean_squared_error(actual, predicted))

st.title("📈 Crypto Price Predictor")
st.markdown("Predice precios de criptomonedas usando **ARIMA** y **Prophet**")

with st.sidebar:
    st.header("Configuración")
    crypto = st.selectbox("Criptomoneda", CRYPTO_LIST, index=0)
    custom = st.text_input("O custom ticker (ej: BNB-USD, TON-USD)")
    if custom:
        crypto = custom.upper().strip()
    col1, col2 = st.columns(2)
    with col1:
        months_back = st.slider("Historial (meses)", 3, 24, 12)
    with col2:
        forecast_days = st.slider("Días a predecir", 7, 90, 30)
    model_choice = st.radio("Modelo", ["Ambos", "ARIMA", "Prophet"])
    run = st.button("🚀 Predecir", type="primary", use_container_width=True)

if run:
    with st.spinner(f"Descargando datos de {crypto}..."):
        period_str = f"{months_back}mo"
        df = fetch_data(crypto, period=period_str)
    if df.empty:
        st.error(f"No se encontraron datos para {crypto}. Verifica el ticker.")
        st.stop()
    series = prepare_series(df)
    last_date = series["ds"].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
    tab1, tab2, tab3 = st.tabs(["📊 Precios Históricos", "🔮 Predicciones", "📋 Datos"])
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"],
            mode="lines", name="Close Price",
            line=dict(color="#00d4aa", width=2)
        ))
        if "Volume" in df.columns:
            fig.add_trace(go.Bar(
                x=df.index, y=df["Volume"],
                name="Volume", yaxis="y2",
                marker_color="rgba(0, 212, 170, 0.15)"
            ))
        fig.update_layout(
            title=f"{crypto} — Precio Histórico",
            xaxis_title="Fecha",
            yaxis_title="Precio (USD)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)
        col1, col2, col3, col4 = st.columns(4)
        current_price = df["Close"].iloc[-1]
        price_change = df["Close"].iloc[-1] - df["Close"].iloc[0]
        change_pct = (price_change / df["Close"].iloc[0]) * 100
        high = df["Close"].max()
        low = df["Close"].min()
        col1.metric("Precio Actual", f"${current_price:.2f}")
        col2.metric("Cambio Período", f"${price_change:+.2f}", f"{change_pct:+.2f}%")
        col3.metric("Máximo", f"${high:.2f}")
        col4.metric("Mínimo", f"${low:.2f}")
    with tab2:
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = {}
        if model_choice in ["ARIMA", "Ambos"]:
            status_text.text("Entrenando ARIMA...")
            try:
                arima_fitted, arima_forecast = train_arima(series, forecast_days)
                results["ARIMA"] = arima_forecast
            except Exception as e:
                st.error(f"ARIMA falló: {e}")
            progress_bar.progress(50)
        if model_choice in ["Prophet", "Ambos"]:
            status_text.text("Entrenando Prophet...")
            try:
                prophet_model, prophet_forecast = train_prophet(series, forecast_days)
                results["Prophet"] = prophet_forecast
            except Exception as e:
                st.error(f"Prophet falló: {e}")
            progress_bar.progress(100)
            status_text.text("")
        if not results:
            st.warning("No se pudo entrenar ningún modelo.")
            st.stop()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=series["ds"], y=series["y"],
            mode="lines", name="Histórico",
            line=dict(color="#00d4aa", width=2)
        ))
        colors = {"ARIMA": "#ff6b6b", "Prophet": "#4ecdc4"}
        metrics_data = []
        for name, forecast in results.items():
            forecast_dates = future_dates[:len(forecast)]
            if name == "Prophet":
                yhat = forecast.loc[forecast["ds"] >= last_date, "yhat"].values[:forecast_days]
                yhat_lower = forecast.loc[forecast["ds"] >= last_date, "yhat_lower"].values[:forecast_days]
                yhat_upper = forecast.loc[forecast["ds"] >= last_date, "yhat_upper"].values[:forecast_days]
            else:
                yhat = forecast["yhat"].values
                yhat_lower = None
                yhat_upper = None
            fig2.add_trace(go.Scatter(
                x=forecast_dates, y=yhat,
                mode="lines", name=f"{name} (Predicción)",
                line=dict(color=colors.get(name, "#888"), width=2, dash="dash"),
            ))
            if yhat_lower is not None and yhat_upper is not None:
                fig2.add_trace(go.Scatter(
                    x=pd.concat([forecast_dates.to_series(), forecast_dates.to_series()[::-1]]),
                    y=pd.concat([pd.Series(yhat_upper), pd.Series(yhat_lower)[::-1]]),
                    fill="toself",
                    fillcolor=f"rgba{tuple(int(colors.get(name, '#888').lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}",
                    line=dict(width=0),
                    name=f"{name} (IC 80%)",
                    showlegend=True,
                ))
            train_actual = series["y"].values[-min(30, len(series)):]
            if name == "Prophet":
                prophet_pred = forecast.loc[
                    forecast["ds"].isin(series["ds"]), "yhat"
                ].values[-min(30, len(series)):]
                train_pred = prophet_pred if len(prophet_pred) == len(train_actual) else train_actual[:len(prophet_pred)]
            else:
                arima_pred = arima_fitted["yhat"].values[-min(30, len(series)):]
                train_pred = arima_pred if len(arima_pred) == len(train_actual) else train_actual[:len(arima_pred)]
            if len(train_actual) == len(train_pred):
                mae, rmse = compute_metrics(train_actual, train_pred)
                metrics_data.append({
                    "Modelo": name,
                    "MAE": f"${mae:.4f}",
                    "RMSE": f"${rmse:.4f}",
                    "Predicción (fin)": f"${yhat[-1]:.2f}" if len(yhat) > 0 else "N/A",
                })
        fig2.update_layout(
            title=f"{crypto} — Predicción de Precio ({forecast_days} días)",
            xaxis_title="Fecha",
            yaxis_title="Precio (USD)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500,
        )
        st.plotly_chart(fig2, use_container_width=True)
        if metrics_data:
            st.subheader("📊 Métricas de Error (últimos 30 días)")
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    with tab3:
        st.subheader("Datos Históricos")
        display_df = df.copy()
        if "Volume" in display_df.columns:
            display_df["Volume"] = display_df["Volume"].apply(
                lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A"
            )
        st.dataframe(
            display_df.style.format(
                {col: "${:.2f}" for col in ["Open", "High", "Low", "Close"] if col in display_df.columns}
            ),
            use_container_width=True,
        )
        if results:
            st.subheader("Predicciones")
            pred_rows = []
            for i, date in enumerate(future_dates):
                row = {"Fecha": date.strftime("%Y-%m-%d")}
                for name, forecast in results.items():
                    if name == "Prophet":
                        vals = forecast.loc[forecast["ds"] >= last_date, "yhat"].values
                        row[name] = f"${vals[i]:.2f}" if i < len(vals) else "N/A"
                    else:
                        vals = forecast["yhat"].values
                        row[name] = f"${vals[i]:.2f}" if i < len(vals) else "N/A"
                pred_rows.append(row)
            st.dataframe(
                pd.DataFrame(pred_rows),
                use_container_width=True,
                hide_index=True,
            )
        csv = df.to_csv()
        st.download_button(
            label="📥 Descargar datos históricos (CSV)",
            data=csv,
            file_name=f"{crypto}_historico.csv",
            mime="text/csv",
        )
else:
    st.info("Configura los parámetros en la barra lateral y presiona **🚀 Predecir**")
    st.markdown("""
    ### Cómo funciona
    1. **Selecciona** una criptomoneda de la lista o ingresa un ticker personalizado
    2. **Ajusta** el historial y los días a predecir
    3. **Elige** entre ARIMA, Prophet, o ambos para comparar
    4. **Presiona Predecir** y obtén los resultados con gráficos interactivos
    ### Modelos
    - **ARIMA** — Modelo estadístico autorregresivo. Ideal para patrones lineales.
    - **Prophet** — Modelo de Meta. Maneja estacionalidad y cambios de tendencia.
    ### Nota
    Las predicciones son estimaciones basadas en datos históricos. No son recomendaciones de inversión.
    """)
