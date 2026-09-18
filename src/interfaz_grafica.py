
import streamlit as st
import pandas as pd
 
# -----------------------------------------------------------------
# Configuración de la página
# -----------------------------------------------------------------
st.set_page_config(
    page_title="Tablero Spaceship Titanic",
    layout="wide",
)
 
df = pd.read_csv("../data/titanic_data/train.csv")
 
# Deck: la primera letra de Cabin (B/0/P -> "B"). Cabin tiene demasiados
# valores únicos para graficarla directo, pero el deck sí es una buena
# categoría (equivalente a "Colonia" en el ejemplo de Ames).
df['Deck'] = df['Cabin'].str.split('/').str[0]
 
df['AgeRange'] = pd.cut(
    df['Age'], bins=[0, 18, 30, 50, 100],
    labels=['0-18', '19-30', '31-50', '51+']
)
 
# -----------------------------------------------------------------
# Encabezado
# -----------------------------------------------------------------
st.title("Tablero de pasajeros — Spaceship Titanic")
st.caption(f"{df.shape[0]:,} registros en el alcance actual")
 
# -----------------------------------------------------------------
# Filtros (en la barra lateral, y que SÍ filtran el df)
# -----------------------------------------------------------------
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
 
# -----------------------------------------------------------------
# KPIs
# -----------------------------------------------------------------
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
 
# -----------------------------------------------------------------
# Gráficos categóricos
# -----------------------------------------------------------------
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
 
# -----------------------------------------------------------------
# Gráficos ordinales
# -----------------------------------------------------------------
st.subheader("Tasa de transportados por variable ordinal")
 
o1, o2 = st.columns(2)
 
with o1:
    st.markdown("**CryoSleep**")
    st.bar_chart(df_filtered.groupby('CryoSleep')['Transported'].mean())
 
with o2:
    st.markdown("**Rango de edad**")
    st.bar_chart(df_filtered.groupby('AgeRange', observed=True)['Transported'].mean())
 
st.divider()
 
# -----------------------------------------------------------------
# Tabla de registros
# -----------------------------------------------------------------
st.subheader("Registros")
st.caption(f"Mostrando {min(20, df_filtered.shape[0])} de {df_filtered.shape[0]:,} pasajeros que cumplen el filtro")
 
columnas_tabla = [
    'PassengerId', 'HomePlanet', 'CryoSleep', 'Deck', 'Destination',
    'Age', 'VIP', 'Transported'
]
st.dataframe(df_filtered[columnas_tabla].head(20), use_container_width=True, hide_index=True)