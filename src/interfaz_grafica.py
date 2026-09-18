#type: ignore

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Tablero Spaceship Titanic",
    layout="wide",
)

df = pd.read_csv("../data/titanic_data/train.csv")

# -----------------------------------------------------------------
# Preprocesamiento: pipeline_preprocesamiento.pkl NO es un Pipeline de
# sklearn, es un diccionario de estadísticas (medianas, modas, el
# StandardScaler ya ajustado) que alimenta esta función. Es la MISMA
# función usada al entrenar (sección 19.1 del notebook) — hay que
# aplicarla tal cual sobre cualquier dato nuevo.
# -----------------------------------------------------------------
def preparar_features(df_raw, stats=None):
    df = df_raw.copy()

    # 2.3 / 3.1: separar Cabin en Deck, NumCabin y Side
    df[['Deck', 'NumCabin', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['NumCabin'] = pd.to_numeric(df['NumCabin'])

    # 3.2: descarte de variables (PassengerId se maneja fuera de esta función)
    df = df.drop(columns=['Name', 'Age', 'VIP', 'Cabin'])

    ajustando = stats is None
    if ajustando:
        stats = {}

    # 4.1: imputación informativa de las variables de gasto, condicionada a CryoSleep
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    gasto_total = df[spend_cols].sum(axis=1, skipna=True)
    dormido = df['CryoSleep'] == True
    despierto_o_faltante = df['CryoSleep'].fillna(False) == False

    if ajustando:
        stats['mediana_gasto'] = {
            col: df.loc[despierto_o_faltante & df[col].notna(), col].median()
            for col in spend_cols
        }
    for col in spend_cols:
        falta = df[col].isna()
        df.loc[falta & dormido, col] = 0
        df.loc[falta & despierto_o_faltante, col] = stats['mediana_gasto'][col]

    # 4.2: imputación de faltantes reales
    falta_cryo = df['CryoSleep'].isna()
    df.loc[falta_cryo & (gasto_total > 0), 'CryoSleep'] = False
    if ajustando:
        stats['moda_cryosleep'] = df['CryoSleep'].mode()[0]
    df['CryoSleep'] = df['CryoSleep'].fillna(stats['moda_cryosleep'])

    def moda_por_grupo(serie):
        return serie.mode().iloc[0] if serie.notna().any() else np.nan

    if ajustando:
        stats['homeplanet_por_deck'] = df.groupby('Deck')['HomePlanet'].apply(moda_por_grupo)
        stats['moda_homeplanet'] = df['HomePlanet'].mode()[0]
    df['HomePlanet'] = df['HomePlanet'].fillna(df['Deck'].map(stats['homeplanet_por_deck']))
    df['HomePlanet'] = df['HomePlanet'].fillna(stats['moda_homeplanet'])

    if ajustando:
        stats['moda_destination'] = df['Destination'].mode()[0]
    df['Destination'] = df['Destination'].fillna(stats['moda_destination'])

    if ajustando:
        stats['deck_por_homeplanet'] = df.groupby('HomePlanet')['Deck'].apply(moda_por_grupo)
        stats['moda_deck'] = df['Deck'].mode()[0]
    df['Deck'] = df['Deck'].fillna(df['HomePlanet'].map(stats['deck_por_homeplanet']))
    df['Deck'] = df['Deck'].fillna(stats['moda_deck'])

    if ajustando:
        stats['moda_side'] = df['Side'].mode()[0]
    df['Side'] = df['Side'].fillna(stats['moda_side'])

    if ajustando:
        stats['numcabin_por_deck'] = df.groupby('Deck')['NumCabin'].median()
        stats['mediana_numcabin'] = df['NumCabin'].median()
    df['NumCabin'] = df['NumCabin'].fillna(df['Deck'].map(stats['numcabin_por_deck']))
    df['NumCabin'] = df['NumCabin'].fillna(stats['mediana_numcabin'])

    # 3.3: binarización de las variables de gasto
    for col in spend_cols:
        df[f'Uso{col}'] = (df[col] > 0).astype(int)
    df = df.drop(columns=spend_cols)

    # 5.1: codificación nominal
    df = pd.get_dummies(df, columns=['HomePlanet', 'Destination', 'Deck'], drop_first=True)
    df['Side'] = df['Side'].map({'P': 0, 'S': 1})
    df['CryoSleep'] = df['CryoSleep'].astype(int)

    # 5.2: codificación ordinal de NumCabin
    if ajustando:
        stats['bins_numcabin'] = [0, 300, 600, 900, 1200, 1500, 1800, np.inf]
    df['NumCabin_bin'] = pd.cut(df['NumCabin'], bins=stats['bins_numcabin'],
                                 labels=False, include_lowest=True)

    # 7.2: escalado
    v_escalar = ['NumCabin', 'NumCabin_bin']
    if ajustando:
        stats['scaler'] = StandardScaler()
        df[v_escalar] = stats['scaler'].fit_transform(df[v_escalar])
    else:
        df[v_escalar] = stats['scaler'].transform(df[v_escalar])

    return df, stats

PIPELINE_PATH = "../models/pipeline_preprocesamiento.pkl"
MODEL_PATH = "../models/modelo_final_arbol_decision.pkl"

@st.cache_resource
def cargar_stats():
    try:
        return joblib.load(PIPELINE_PATH)
    except FileNotFoundError:
        return None

@st.cache_resource
def cargar_modelo():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        return None

stats_pipeline = cargar_stats()
modelo = cargar_modelo()

# Deck: la primera letra de Cabin (B/0/P -> "B"). Cabin tiene demasiados
# valores únicos para graficarla directo, pero el deck sí es una buena
# categoría (equivalente a "Colonia" en el ejemplo de Ames).
df['Deck'] = df['Cabin'].str.split('/').str[0]
df['NumCabin'] = pd.to_numeric(df['Cabin'].str.split('/').str[1], errors='coerce')
 
# Rango de números de cabina positivos observado por Deck en train.csv para
# limitar el formulario a los valores aceptados por la aplicación.
rangos_numcabin_por_deck = (
        df.loc[df['NumCabin'] > 0]
            .groupby('Deck')['NumCabin']
            .agg(['min', 'max'])
)
 
df['AgeRange'] = pd.cut(
    df['Age'], bins=[0, 18, 30, 50, 100],
    labels=['0-18', '19-30', '31-50', '51+']
)

st.title("Tablero de pasajeros — Spaceship Titanic")
st.caption(f"{df.shape[0]:,} registros en el alcance actual")

st.sidebar.header("Filtros")

homeplanet_options = ["Todos"] + sorted(df['HomePlanet'].dropna().unique().tolist())
destination_options = ["Todos"] + sorted(df['Destination'].dropna().unique().tolist())

homeplanet_sel = st.sidebar.selectbox("HomePlanet", homeplanet_options)
destination_sel = st.sidebar.selectbox("Destination", destination_options)

df_filtered = df.copy()
if homeplanet_sel != "Todos":
    df_filtered = df_filtered[df_filtered['HomePlanet'] == homeplanet_sel]
if destination_sel != "Todos":
    df_filtered = df_filtered[df_filtered['Destination'] == destination_sel]

st.sidebar.caption(f"{df_filtered.shape[0]:,} de {df.shape[0]:,} pasajeros cumplen el filtro")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Pasajeros", f"{df_filtered.shape[0]:,}")

if df_filtered.shape[0] > 0:
    tasa_transportados = df_filtered['Transported'].mean() * 100
    edad_promedio = df_filtered['Age'].mean()
    gasto_promedio = df_filtered[
        ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    ].sum(axis=1).mean()
else:
    tasa_transportados = edad_promedio = gasto_promedio = 0

col2.metric("% Transportados", f"{tasa_transportados:.1f}%")
col3.metric("Edad promedio", f"{edad_promedio:.1f} años")
col4.metric("Gasto promedio total", f"${gasto_promedio:,.2f}")

st.divider()

st.subheader("Tasa de transportados por categoría")

g1, g2, g3 = st.columns(3)

with g1:
    st.markdown("**Por HomePlanet**")
    st.bar_chart(df_filtered.groupby('HomePlanet')['Transported'].mean().sort_values(ascending=False))

with g2:
    st.markdown("**Por Destination**")
    st.bar_chart(df_filtered.groupby('Destination')['Transported'].mean().sort_values(ascending=False))

with g3:
    st.markdown("**Por Deck (Cabin)**")
    st.bar_chart(df_filtered.groupby('Deck')['Transported'].mean().sort_values(ascending=False))

st.divider()

st.subheader("Tasa de transportados por variable ordinal")

o1, o2 = st.columns(2)

with o1:
    st.markdown("**CryoSleep**")
    st.bar_chart(df_filtered.groupby('CryoSleep')['Transported'].mean())

with o2:
    st.markdown("**Rango de edad**")
    st.bar_chart(df_filtered.groupby('AgeRange', observed=True)['Transported'].mean())

st.divider()

st.subheader("Registros")
st.caption(f"Mostrando {min(20, df_filtered.shape[0])} de {df_filtered.shape[0]:,} pasajeros que cumplen el filtro")

columnas_tabla = [
    'PassengerId', 'HomePlanet', 'CryoSleep', 'Deck', 'Destination',
    'Age', 'VIP', 'Transported'
]
st.dataframe(df_filtered[columnas_tabla].head(20), use_container_width=True, hide_index=True)

st.divider()

st.subheader("Predicción para un pasajero")

if modelo is None or stats_pipeline is None:
    faltantes = []
    if stats_pipeline is None:
        faltantes.append(f"`{STATS_PATH}`")
    if modelo is None:
        faltantes.append(f"`{MODEL_PATH}`")
    st.warning(
        f"No encontré {' ni '.join(faltantes)} en esta carpeta. "
        "Colócalos junto a `app.py`."
    )
else:
    deck_options = sorted(df['Deck'].dropna().unique())
    deck = st.selectbox("Deck", deck_options, key="deck_prediccion")
 
    rango_deck = rangos_numcabin_por_deck.loc[deck]
    max_cabin_observada = int(rango_deck['max'])
    st.caption(
        f"Rango observado en train para el Deck {deck}: 1 – {max_cabin_observada}. "
        "Se acepta cualquier número de cabina positivo."
    )
    cryo_sleep = st.checkbox("CryoSleep", key="cryosleep_prediccion")
    if cryo_sleep:
        st.caption("Los campos de gasto se deshabilitan: un pasajero dormido no puede gastar.")

    with st.form("form_prediccion"):
        c1, c2, c3 = st.columns(3)
 
        with c1:
            home_planet = st.selectbox("HomePlanet", sorted(df['HomePlanet'].dropna().unique()))
            destination = st.selectbox("Destination", sorted(df['Destination'].dropna().unique()))
 
        with c2:
            st.markdown(f"**Deck**: {deck}")
            num_cabin = st.number_input(
                "Número de cabina",
                min_value=1,
                value=max(1, max_cabin_observada // 2),
                key=f"num_cabin_{deck}",
            )
            side = st.selectbox("Side", ["P", "S"])
 
        with c3:
            sufijo = "dormido" if cryo_sleep else "despierto"
            room_service = st.number_input("RoomService", min_value=0, value=0, disabled=cryo_sleep, key=f"room_service_{sufijo}")
            food_court = st.number_input("FoodCourt", min_value=0, value=0, disabled=cryo_sleep, key=f"food_court_{sufijo}")
            shopping_mall = st.number_input("ShoppingMall", min_value=0, value=0, disabled=cryo_sleep, key=f"shopping_mall_{sufijo}")
            spa = st.number_input("Spa", min_value=0, value=0, disabled=cryo_sleep, key=f"spa_{sufijo}")
            vr_deck = st.number_input("VRDeck", min_value=0, value=0, disabled=cryo_sleep, key=f"vr_deck_{sufijo}")
 
        enviar = st.form_submit_button("Predecir")

    if enviar:
        if num_cabin <= 0:
            st.error("El número de cabina debe ser mayor que 0.")
            st.stop()

        # Respaldo de seguridad: si CryoSleep está activo, el gasto SIEMPRE
        # es 0, sin importar lo que hayan quedado los widgets deshabilitados.
        if cryo_sleep:
            room_service = food_court = shopping_mall = spa = vr_deck = 0

        # Nota: 'Name', 'Age' y 'VIP' no influyen en la predicción (preparar_features
        # los descarta), pero deben existir como columnas para que el .drop() no truene.
        input_data = pd.DataFrame([{
            'Name': 'Desconocido Desconocido',
            'Age': 0,
            'VIP': False,
            'HomePlanet': home_planet,
            'CryoSleep': cryo_sleep,
            'Cabin': f"{deck}/{num_cabin}/{side}",
            'Destination': destination,
            'RoomService': room_service,
            'FoodCourt': food_court,
            'ShoppingMall': shopping_mall,
            'Spa': spa,
            'VRDeck': vr_deck,
        }])

        try:
            # 1. Aplicar EXACTAMENTE la misma función de preprocesamiento del entrenamiento
            input_procesado, _ = preparar_features(input_data, stats=stats_pipeline)

            # 2. Como get_dummies sobre una sola fila puede no generar todas las
            # columnas que el modelo vio al entrenar (p. ej. si esta fila no cae en
            # esa categoría), reindexamos contra las columnas que el modelo espera.
            if hasattr(modelo, "feature_names_in_"):
                input_procesado = input_procesado.reindex(
                    columns=modelo.feature_names_in_, fill_value=0
                )

            # 3. Predecir con el modelo ya entrenado
            prediccion = modelo.predict(input_procesado)[0]

            r1, r2 = st.columns(2)

            if prediccion:
                r1.success("Transportado")
            else:
                r1.error("No transportado")

            if hasattr(modelo, "predict_proba"):
                proba = modelo.predict_proba(input_procesado)[0]
                clase_positiva = list(modelo.classes_).index(True) if True in modelo.classes_ else 1
                r2.metric("Pasajeros similares en train.csv que fueron transportados", f"{proba[clase_positiva] * 100:.1f}%")

        except Exception as e:
            st.error(
                "Algo falló al preprocesar o predecir. Revisa el traceback abajo — "
                "es probable que sea un desfase de columnas entre `preparar_features` "
                "y lo que espera tu modelo."
            )
            st.exception(e)