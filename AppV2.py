import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Data Engine Pro", layout="wide")

# --- FUNCIONES DE LÓGICA (BACKEND) ---


@st.cache_data
def ingestar_datos(archivo):
    """Detecta delimitador y maneja errores de carga."""
    try:
        delimitadores = [",", ";", "\t"]
        for sep in delimitadores:
            archivo.seek(0)
            df = pd.read_csv(archivo, sep=sep, nrows=5)
            if df.shape[1] > 1:
                archivo.seek(0)
                return pd.read_csv(archivo, sep=sep)
        archivo.seek(0)
        return pd.read_csv(archivo)  # Fallback a coma
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None


def panel_limpieza(df_raw):
    """Panel interactivo para limpiar datos."""
    df = df_raw.copy()
    with st.sidebar.expander("🛠️ Panel de Limpieza de Datos", expanded=False):
        # Conversión de fechas automática
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    df[col] = pd.to_datetime(df[col], errors="ignore")
                except:
                    pass

        # Manejo de Nulos
        st.write("Tratamiento de valores nulos:")
        metodo_nulos = st.radio(
            "Acción:",
            ["Mantener", "Eliminar Filas", "Imputar Media/Moda"],
            label_visibility="collapsed",
        )

        if metodo_nulos == "Eliminar Filas":
            df = df.dropna()
        elif metodo_nulos == "Imputar Media/Moda":
            for col in df.columns:
                if df[col].dtype in ["float64", "int64"]:
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna(
                        df[col].mode()[0] if not df[col].mode().empty else "Desconocido"
                    )
    return df


def obtener_sugerencia_grafico(df):
    """Sugiere el mejor gráfico basado en tipos de datos."""
    cols_num = df.select_dtypes(include=["number"]).columns.tolist()
    cols_date = df.select_dtypes(include=["datetime"]).columns.tolist()

    if cols_date and cols_num:
        return "Líneas", cols_date[0], cols_num[0]
    elif len(cols_num) >= 2:
        return "Dispersión", cols_num[0], cols_num[1]
    elif len(cols_num) == 1 and len(df.columns) > 1:
        return "Barras", df.columns[0], cols_num[0]
    else:
        return "Dispersión", df.columns[0], df.columns[-1]


def entrenar_modelo_ml(df, target, features, tipo_modelo):
    """Entrena un modelo de ML y devuelve métricas y resultados."""
    # Preparamos los datos (solo numéricos y sin nulos para ML)
    df_ml = df[features + [target]].dropna()

    X = df_ml[features]
    y = df_ml[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if tipo_modelo == "Regresión Lineal":
        modelo = LinearRegression()
    else:
        modelo = RandomForestRegressor(n_estimators=100, random_state=42)

    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)

    r2 = r2_score(y_test, predicciones)
    mae = mean_absolute_error(y_test, predicciones)

    # DataFrame para graficar
    df_resultados = pd.DataFrame({"Valor Real": y_test, "Predicción": predicciones})

    return r2, mae, df_resultados


# --- INTERFAZ DE USUARIO (FRONTEND) ---

st.title("🥼 Asistente de Labo v3.0 🧪")
st.markdown("*Plataforma integral de ingesta, análisis predictivo y visualización.*")

st.sidebar.header("1. Ingesta de Datos")
archivo_subido = st.sidebar.file_uploader("Subir CSV", type=["csv"])

if archivo_subido:
    df_raw = ingestar_datos(archivo_subido)

    if df_raw is not None and not df_raw.empty:
        # 1. Aplicamos Limpieza
        df_limpio = panel_limpieza(df_raw)

        # 2. Filtros Dinámicos
        st.sidebar.markdown("---")
        st.sidebar.subheader("2. Filtros Dinámicos")
        columnas_cat = df_limpio.select_dtypes(include=["object", "category"]).columns
        df_filtrado = df_limpio.copy()

        if len(columnas_cat) > 0:
            col_f = st.sidebar.selectbox("Filtrar por:", columnas_cat)
            valores_unicos = df_limpio[col_f].unique()
            opciones = st.sidebar.multiselect(
                f"Valores de {col_f}", valores_unicos, default=valores_unicos
            )
            df_filtrado = df_limpio[df_limpio[col_f].isin(opciones)]

        # --- SISTEMA DE PESTAÑAS ---
        tab1, tab2, tab3 = st.tabs(
            ["📊 Exploración", "📈 Análisis Avanzado", "🤖 Machine Learning"]
        )

        # PESTAÑA 1: EXPLORACIÓN BÁSICA
        with tab1:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Métricas y Datos")
                st.metric("Filas post-filtro", df_filtrado.shape[0])
                st.dataframe(df_filtrado.head(15), use_container_width=True)

                # Descarga de CSV procesado
                csv = df_filtrado.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Descargar CSV Procesado",
                    data=csv,
                    file_name="datos_limpios.csv",
                    mime="text/csv",
                )

            with col2:
                if not df_filtrado.empty:
                    st.subheader("Visualización Inteligente")
                    sug_tipo, sug_x, sug_y = obtener_sugerencia_grafico(df_filtrado)

                    c1, c2, c3 = st.columns(3)
                    opciones_graf = ["Barras", "Líneas", "Dispersión"]
                    idx_tipo = (
                        opciones_graf.index(sug_tipo)
                        if sug_tipo in opciones_graf
                        else 0
                    )

                    tipo = c1.selectbox(
                        "Tipo de Gráfico", opciones_graf, index=idx_tipo
                    )

                    # Evitar errores si cambian las columnas drásticamente
                    idx_x = (
                        list(df_filtrado.columns).index(sug_x)
                        if sug_x in df_filtrado.columns
                        else 0
                    )
                    idx_y = (
                        list(df_filtrado.columns).index(sug_y)
                        if sug_y in df_filtrado.columns
                        else min(1, len(df_filtrado.columns) - 1)
                    )

                    x_axis = c2.selectbox("Eje X", df_filtrado.columns, index=idx_x)
                    y_axis = c3.selectbox("Eje Y", df_filtrado.columns, index=idx_y)

                    if tipo == "Barras":
                        fig = px.bar(
                            df_filtrado, x=x_axis, y=y_axis, template="plotly_dark"
                        )
                    elif tipo == "Líneas":
                        fig = px.line(
                            df_filtrado, x=x_axis, y=y_axis, template="plotly_dark"
                        )
                    else:
                        fig = px.scatter(
                            df_filtrado,
                            x=x_axis,
                            y=y_axis,
                            template="plotly_dark",
                            opacity=0.7,
                        )

                    st.plotly_chart(fig, use_container_width=True)

        # PESTAÑA 2: ESTADÍSTICA AVANZADA
        with tab2:
            st.subheader("Estadística Descriptiva y Correlación")
            st.dataframe(df_filtrado.describe().T, use_container_width=True)

            df_num = df_filtrado.select_dtypes(include=["number"])
            if not df_num.empty and df_num.shape[1] > 1:
                st.markdown("##### Matriz de Correlación (Heatmap)")
                corr = df_num.corr()
                fig_corr = px.imshow(
                    corr,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    template="plotly_dark",
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info(
                    "💡 Se necesitan al menos 2 columnas numéricas para calcular la matriz de correlación."
                )

        # PESTAÑA 3: MACHINE LEARNING
        with tab3:
            st.subheader("🤖 Laboratorio Predictivo")
            st.markdown(
                "Entrena un modelo para predecir el comportamiento de una variable basándote en el resto de los datos numéricos."
            )

            df_ml = df_filtrado.select_dtypes(include=["number"])

            if df_ml.shape[1] >= 2 and len(df_ml) >= 20:
                col_ml1, col_ml2 = st.columns([1, 2])

                with col_ml1:
                    target = st.selectbox(
                        "🎯 ¿Qué quieres predecir? (Variable Y)",
                        df_ml.columns,
                        index=len(df_ml.columns) - 1,
                    )
                    features_disp = [c for c in df_ml.columns if c != target]
                    features = st.multiselect(
                        "📊 ¿Qué variables usarás para predecir? (Variables X)",
                        features_disp,
                        default=features_disp,
                    )

                    tipo_modelo = st.radio(
                        "⚙️ Algoritmo:",
                        ["Regresión Lineal", "Random Forest (Más complejo)"],
                    )

                    entrenar = st.button("🚀 Entrenar Modelo", use_container_width=True)

                with col_ml2:
                    if entrenar:
                        if len(features) > 0:
                            with st.spinner("Entrenando modelo..."):
                                r2, mae, df_res = entrenar_modelo_ml(
                                    df_filtrado, target, features, tipo_modelo
                                )

                                st.success("¡Modelo entrenado con éxito!")
                                c_met1, c_met2 = st.columns(2)
                                c_met1.metric(
                                    label="Coeficiente de Determinación ($R^2$)",
                                    value=f"{r2:.3f}",
                                    help="Más cerca de 1.0 es mejor.",
                                )
                                c_met2.metric(
                                    label="Error Absoluto Medio (MAE)",
                                    value=f"{mae:.2f}",
                                    help="En las mismas unidades que tu variable a predecir.",
                                )

                                # Gráfico de Real vs Predicción

                                fig_ml = px.scatter(
                                    df_res,
                                    x="Valor Real",
                                    y="Predicción",
                                    title="Precisión del Modelo: Real vs. Predicho",
                                    template="plotly_dark",
                                    trendline="ols",
                                )
                                st.plotly_chart(fig_ml, use_container_width=True)
                        else:
                            st.warning(
                                "⚠️ Debes seleccionar al menos una variable predictora (X)."
                            )
            else:
                st.info(
                    "Para usar Machine Learning necesitas al menos 20 filas de datos numéricos y 2 columnas numéricas."
                )

    else:
        st.error("El archivo está vacío o corrupto. Por favor revisa tu CSV.")
else:
    st.info("👋 Sube un archivo CSV desde la barra lateral para encender los motores.")
