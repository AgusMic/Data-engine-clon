import streamlit as st
import pandas as pd
import plotly.express as px

# Configuración de la página
st.set_page_config(page_title="Data Analysis Platform", layout="wide")

st.title("🛡️ Data Engine v1.0")
st.markdown("---")

# --- MÓDULO 1: INGESTA DE DATOS ---
st.sidebar.header("Configuración de Datos")
archivo_subido = st.sidebar.file_uploader("Cargar fuente de datos (CSV)", type=["csv"])

if archivo_subido is not None:
    df = pd.read_csv(archivo_subido)

    # --- MÓDULO 2: FILTROS DINÁMICOS ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filtros de Análisis")

    columnas_categoricas = df.select_dtypes(include=["object"]).columns

    if len(columnas_categoricas) > 0:
        col_filtro = st.sidebar.selectbox("Filtrar por:", columnas_categoricas)
        valores = df[col_filtro].unique()
        seleccion = st.sidebar.multiselect(
            f"Selecciona {col_filtro}", valores, default=valores
        )
        # 🟢 CAMBIO IMPORTANTE: Creamos el DataFrame filtrado
        df_filtrado = df[df[col_filtro].isin(seleccion)]
    else:
        df_filtrado = df

    # --- MÓDULO 3: EXPLORACIÓN Y MÉTRICAS ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Métricas Clave")
        st.metric("Total Registros", len(df_filtrado))

        columnas_numericas = df_filtrado.select_dtypes(include=["number"]).columns
        if columnas_numericas.any():
            col_num = columnas_numericas[0]
            st.metric(f"Promedio {col_num}", round(df_filtrado[col_num].mean(), 2))

        st.write("Vista previa:")
        st.dataframe(df_filtrado.head(10), use_container_width=True)

    # --- MÓDULO 4: VISUALIZACIÓN ---
    with col2:
        st.subheader("Análisis Visual")
        # 🟢 CAMBIO: Usamos df_filtrado.columns para los selectores
        eje_x = st.selectbox("Eje X", df_filtrado.columns)
        eje_y = st.selectbox("Eje Y", df_filtrado.columns)
        tipo_grafico = st.radio(
            "Tipo de gráfico", ["Barras", "Líneas", "Dispersión"], horizontal=True
        )

        # 🟢 CAMBIO CLAVE: Usamos df_filtrado en los gráficos para que respondan a los filtros
        if tipo_grafico == "Barras":
            fig = px.bar(df_filtrado, x=eje_x, y=eje_y, template="plotly_dark")
        elif tipo_grafico == "Líneas":
            fig = px.line(df_filtrado, x=eje_x, y=eje_y, template="plotly_dark")
        else:
            fig = px.scatter(df_filtrado, x=eje_x, y=eje_y, template="plotly_dark")

        st.plotly_chart(fig, use_container_width=True)
else:
    st.info(
        "👋 Bienvenido. Por favor, sube un archivo CSV desde la barra lateral para comenzar."
    )
