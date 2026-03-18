import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import statsmodels.api as sm
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from supabase import create_client, Client

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="DF Analytics", layout="wide")


# --- CONEXIÓN A LA BASE DE DATOS (SUPABASE) ---
@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


try:
    supabase = init_connection()
except Exception as e:
    st.error(f"Error conectando a la base de datos: {e}")
    st.stop()

# --- GESTIÓN DE SESIÓN (MEMORIA) ---
if "usuario" not in st.session_state:
    st.session_state["usuario"] = None


# --- SISTEMA DE LOGIN Y REGISTRO ---
def mostrar_login():
    st.title("🔐 Acceso")
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        modo = st.radio(
            "Selecciona una opción:",
            ["Iniciar Sesión", "Crear Cuenta nueva"],
            horizontal=True,
        )
        email = st.text_input("Correo electrónico (Email)")
        password = st.text_input("Contraseña", type="password")

        if st.button("Continuar", use_container_width=True):
            if modo == "Crear Cuenta nueva":
                try:
                    response = supabase.auth.sign_up(
                        {"email": email, "password": password}
                    )
                    st.success(
                        "✅ ¡Cuenta creada exitosamente! Ahora selecciona 'Iniciar Sesión' para entrar."
                    )
                except Exception as e:
                    st.error(f"Error al crear la cuenta: {e}")

            elif modo == "Iniciar Sesión":
                try:
                    response = supabase.auth.sign_in_with_password(
                        {"email": email, "password": password}
                    )
                    st.session_state["usuario"] = response.user.email

                    # Guardamos el registro en tu tabla SQL
                    supabase.table("registros_acceso").insert(
                        {"usuario_email": email}
                    ).execute()

                    # Recargamos la página para que desaparezca el login
                    st.rerun()
                except Exception as e:
                    st.error("❌ Correo o contraseña incorrectos. Inténtalo de nuevo.")


# 🛑 LA BARRERA DE SEGURIDAD 🛑
if st.session_state["usuario"] is None:
    mostrar_login()
    st.stop()  # Si no hay usuario, el código se detiene aquí y no lee lo de abajo.

# =====================================================================
# A PARTIR DE AQUÍ VA EL RESTO DE TU APLICACIÓN
# =====================================================================


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

    df_resultados = pd.DataFrame({"Valor Real": y_test, "Predicción": predicciones})

    return r2, mae, df_resultados


# --- FUNCIONES MATEMÁTICAS PARA AJUSTE ---
def modelo_lineal(x, m, b):
    return m * x + b


def modelo_cuadratico(x, a, b, c):
    return a * x**2 + b * x + c


def modelo_exponencial(x, a, b):
    return a * np.exp(b * x)


# --- INTERFAZ DE USUARIO (FRONTEND) ---

st.title("🥼 Asistente de Labo v3.0 🧪")
st.markdown("*Plataforma integral de ingesta, análisis predictivo y visualización.*")

# --- PERFIL DEL USUARIO ---
st.sidebar.success(f"👤 Conectado como:\n**{st.session_state['usuario']}**")
if st.sidebar.button("Cerrar Sesión"):
    st.session_state["usuario"] = None
    st.rerun()
st.sidebar.markdown("---")

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
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📊 Exploración",
                "📈 Análisis Avanzado",
                "🤖 Machine Learning",
                "📐 Ajuste de Curvas",
            ]
        )

        # PESTAÑA 1: EXPLORACIÓN BÁSICA
        with tab1:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Métricas y Datos")
                st.metric("Filas post-filtro", df_filtrado.shape[0])
                st.dataframe(df_filtrado.head(15), use_container_width=True)

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

        # PESTAÑA 4: AJUSTE DE CURVAS Y TENDENCIAS
        with tab4:
            st.subheader("📐 Ajuste de Curvas Experimentales")
            st.markdown(
                "Encuentra la relación matemática entre dos variables o descubre tendencias ocultas."
            )

            df_num = df_filtrado.select_dtypes(include=["number"]).dropna()

            if df_num.shape[1] >= 2:
                c_ajuste1, c_ajuste2 = st.columns([1, 2])

                with c_ajuste1:
                    x_col = st.selectbox(
                        "Variable Independiente (X)", df_num.columns, index=0
                    )
                    y_col = st.selectbox(
                        "Variable Dependiente (Y)", df_num.columns, index=1
                    )

                    metodo = st.radio(
                        "Enfoque Analítico:",
                        [
                            "Determinista (Física Clásica)",
                            "Inteligente (Suavizado LOWESS)",
                        ],
                    )

                    if metodo == "Determinista (Física Clásica)":
                        funcion_elegida = st.selectbox(
                            "Modelo Teórico:",
                            [
                                "Lineal (y = mx + b)",
                                "Cuadrático (y = ax² + bx + c)",
                                "Exponencial (y = A·e^(Bx))",
                            ],
                        )
                    else:
                        st.info(
                            "💡 LOWESS dibujará la tendencia natural de los datos sin forzar una ecuación preestablecida."
                        )

                    ejecutar_ajuste = st.button(
                        "Aplicar Ajuste", use_container_width=True
                    )

                with c_ajuste2:
                    if ejecutar_ajuste:
                        x_data = df_num[x_col].values
                        y_data = df_num[y_col].values

                        # Ordenar los datos en X es clave para dibujar bien las líneas continuas
                        sort_idx = np.argsort(x_data)
                        x_data = x_data[sort_idx]
                        y_data = y_data[sort_idx]

                        fig_fit = px.scatter(
                            df_num,
                            x=x_col,
                            y=y_col,
                            template="plotly_dark",
                            opacity=0.6,
                            title=f"Ajuste: {y_col} vs {x_col}",
                        )

                        try:
                            if metodo == "Determinista (Física Clásica)":
                                if "Lineal" in funcion_elegida:
                                    popt, _ = curve_fit(modelo_lineal, x_data, y_data)
                                    y_fit = modelo_lineal(x_data, *popt)
                                    st.success(
                                        f"**Parámetros Hallados:** m = {popt[0]:.4f} | b = {popt[1]:.4f}"
                                    )

                                elif "Cuadrático" in funcion_elegida:
                                    popt, _ = curve_fit(
                                        modelo_cuadratico, x_data, y_data
                                    )
                                    y_fit = modelo_cuadratico(x_data, *popt)
                                    st.success(
                                        f"**Parámetros Hallados:** a = {popt[0]:.4f} | b = {popt[1]:.4f} | c = {popt[2]:.4f}"
                                    )

                                elif "Exponencial" in funcion_elegida:
                                    popt, _ = curve_fit(
                                        modelo_exponencial, x_data, y_data
                                    )
                                    y_fit = modelo_exponencial(x_data, *popt)
                                    st.success(
                                        f"**Parámetros Hallados:** A = {popt[0]:.4f} | B = {popt[1]:.4f}"
                                    )

                                # Agregar la línea del modelo físico
                                fig_fit.add_scatter(
                                    x=x_data,
                                    y=y_fit,
                                    mode="lines",
                                    name="Ajuste Teórico",
                                    line=dict(color="red", width=3),
                                )

                            else:
                                # Aplicar IA LOWESS
                                lowess_res = sm.nonparametric.lowess(
                                    y_data, x_data, frac=0.3
                                )
                                y_fit = lowess_res[:, 1]

                                fig_fit.add_scatter(
                                    x=x_data,
                                    y=y_fit,
                                    mode="lines",
                                    name="Tendencia LOWESS",
                                    line=dict(color="yellow", width=3),
                                )
                                st.success(
                                    "Curva de tendencia calculada exitosamente mediante regresión local."
                                )

                            st.plotly_chart(fig_fit, use_container_width=True)

                        except Exception as e:
                            st.error(
                                f"El ajuste falló. A veces los datos no tienen la forma matemática seleccionada. Error técnico: {e}"
                            )
            else:
                st.info(
                    "Necesitas al menos 2 columnas numéricas para realizar un ajuste de curvas."
                )

    else:
        st.error("El archivo está vacío o corrupto. Por favor revisa tu CSV.")
else:
    st.info("👋 Sube un archivo CSV desde la barra lateral para encender los motores.")
