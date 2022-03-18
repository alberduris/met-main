import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from windrose import WindroseAxes
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from datetime import datetime, timedelta, date, time
import math
from modeler.core import Core


st.set_page_config(layout="wide")

hide_streamlit_style = """
            <style>
            footer:after {
                content:'by https://gregoirejan.github.io / Using frost.met.no API'; 
                visibility: visible;
                display: block;
                position: relative;
                #background-color: red;
                padding: 5px;
                top: 2px;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

##############################################################################################################################################
st.sidebar.title('Boya de Racalada metereological app')


# Load stations list in Norway
@st.cache
def load_stations():
    return Core().findstation(country='Norge')
df = load_stations()
# print(df.info())
# print(df.head())

@st.cache
def load_data():
    df = pd.read_csv('data/BoyaDeRecalada_Clean.csv')
    df["date"] = pd.to_datetime(df["date"], unit='ns') # Datetime
    return df

# Filter list by selecting only statoins with air temp + precipitations + wind speed/directoin
ids_basic = list(pd.read_csv('ids_basic_nonan.csv',header=None,skiprows=1)[1])
df_basic = df[df['id'].isin(ids_basic)]
# print(df_basic.info())
# print(df_basic.head())

# Select Municipality and station #@TODO: Quitar cuando mis datos
municipality = "OSLO"
name_station = "OSLO - BLINDERN"

# Prepare map data (HARD)
map_data = {'lon': -3.04, 'lat': 43.64}
map_df = pd.DataFrame(map_data, index=[0])

# Get id for selected station
dfid = list(df_basic[(df_basic.municipality == municipality) & (df_basic.name == name_station)].id)

# Select time range
d1 = st.sidebar.date_input(
    "Select first date",
    (pd.to_datetime('2021-12-31') - timedelta(days=7)).date())
d2 = st.sidebar.date_input(
    "Select last date",
    pd.to_datetime('2021-12-31').date())

#@TODO: Quitar cuando mis datos
# Compupte time range and select appropriate feature accordingly 
timerange=(str(d1)+'/'+str(d2))
if (d2 - d1) < timedelta(days=14):
    elements_temp = 'air_temperature'
    elements_preci = 'sum(precipitation_amount P1D)'
elif (d2 - d1) < timedelta(days=366*1):
    elements_temp = 'mean(air_temperature P1D)'
    elements_preci = 'sum(precipitation_amount P1D)'
elif (d2 - d1) < timedelta(days=366*25):
    elements_temp = 'mean(air_temperature P1M)'
    elements_preci = 'sum(precipitation_amount P1M)'
else:
    elements_temp = 'mean(air_temperature P1Y)'
    elements_preci = 'sum(precipitation_amount P1Y)'

# Get data for the selected station / feature / time range
# @TODO: Hacer una mía que cargue mis datos
@st.cache
def load_features():
    air_temperature = Core().graphelement(source=dfid[0],elements= elements_temp,referencetime= timerange,rollingmean=1)
    precipitation = Core().graphelement(source=dfid[0],elements= elements_preci,referencetime= timerange,rollingmean=1)
    return air_temperature, precipitation

#df_temperature, df_precipitation = load_features()

@st.cache
def load_features(d1, d2):

    df = load_data()
    d1 = pd.to_datetime(d1, unit='ns')
    d2 = pd.to_datetime(d2, unit='ns')


    # Prepare temperature
    df_temperature = df[['date', 'water_temp']]
    df_temperature = df_temperature.loc[(df_temperature['date'] > d1) & (df_temperature['date'] < d2)]
    df_temperature.rename(columns={'date':'referenceTime', 'water_temp':'value'}, inplace=True)


    # Prepare atm pression
    df_precipitation = df[['date', 'atm_pressure']]
    df_precipitation = df_precipitation.loc[(df_precipitation['date'] > d1) & (df_precipitation['date'] < d2)]
    df_precipitation.rename(columns={'date':'referenceTime', 'atm_pressure':'value'}, inplace=True)
    
    return df_temperature, df_precipitation


df_temperature, df_precipitation = load_features(d1, d2)

# Prepare data for plot
df_temperature = df_temperature.set_index('referenceTime')
df_precipitation = df_precipitation.set_index('referenceTime')


print(df_temperature.info())
print(df_temperature.head())

# Prepare plot for temperature and precipitations
fig_combi = make_subplots(specs=[[{"secondary_y": True}]]) #this a one cell subplot
fig_combi.update_layout(title="Climograph",
                template="plotly_white",title_x=0.5,legend=dict(orientation='h'))

trace1 = go.Bar(x=df_precipitation.index,
        y=df_precipitation.value, opacity=0.4,name='Precipitation')

trace2 = go.Scatter(x=df_temperature.index,
        y=df_temperature.value,name='Air Temperature',mode='lines',line=dict(color='red', width=1))

#The first trace is referenced to the default xaxis, yaxis (ie. xaxis='x1', yaxis='y1')
fig_combi.add_trace(trace1, secondary_y=False)

#The second trace is referenced to xaxis='x1'(i.e. 'x1' is common for the two traces) 
#and yaxis='y2' (the right side yaxis)

fig_combi.add_trace(trace2, secondary_y=True)

fig_combi.update_yaxes(#left yaxis
                title= 'ml',showgrid= False, secondary_y=False)
fig_combi.update_yaxes(#right yaxis
                showgrid= True, 
                title= '°C',
                secondary_y=True)

# Load wind data
@st.cache
def load_wind():
    ws = Core().graphelement(source=dfid[0], elements= 'wind_speed',referencetime= timerange,rollingmean=1)
    ws = ws.set_index("referenceTime").resample('10T').mean().value
    wd = Core().graphelement(source=dfid[0], elements= 'wind_from_direction',referencetime= timerange,rollingmean=1)
    wd = wd.set_index("referenceTime").resample('10T').mean().value
    return ws, wd

#@TODO? Quitar condicional y plottear siempre la rosa del viento
# Layout accordingly if windrose is plotted (timerange < 2 month) 
if (d2 - d1) < timedelta(days=31*2):

    # Plot temp and precipitations
    col1, col2 = st.beta_columns((2,1))

    col1.plotly_chart(fig_combi,use_container_width=True)


    # Windrose plot
    ws, wd = load_wind()

    print(ws, wd)
    

    fig_rose = plt.figure()
    rect = [1,1,1,1] 
    wa = WindroseAxes(fig_rose, rect)
    fig_rose.add_axes(wa)
    wa.bar(wd, ws, normed=True, opening=0.8, edgecolor='white')
    wa.set_legend(title='Wind speed (m/s)', loc="lower right", bbox_to_anchor =(1.25, 0))
    wa.set_title('Windrose')

    st.set_option('deprecation.showPyplotGlobalUse', False)
    col2.pyplot(fig_rose)
else:
    # Plot only temp and precipitations
    st.error("Maximum time range for the Windrose and comparison is 2 month.")
    st.plotly_chart(fig_combi,use_container_width=True)

# Plot map
st.map(map_df, zoom=8.5)