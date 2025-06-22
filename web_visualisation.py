import streamlit as st
from orgelpredigt_analysis import Sermon

from collections import Counter

import plotly.express as px

import plotly.graph_objects as go
import pandas as pd

import folium
import json
import re
import os

#########################
##### CHOOSE SERMON #####
#########################

# Get the list of all files in a directory
with open("predigten_übersicht.json", "r", encoding="utf-8") as file: 
    data = json.load(file)

# Ensure all entries have a 'year' key
cleaned = {k: v for k, v in data.items() if 'year' in v}

year_finder = re.compile(r'[0-9]{4}')

for k, v in data.items():
    year = re.findall(year_finder, v['year'])[0]
    if year:
        v['year'] = year
    else:
        v['year'] = '[s.a.]'

# Convert to nested list and sort by year
relevant_sermons = sorted(
    [[key, value['title'], int(value['year'])] for key, value in cleaned.items()],
    key=lambda x: x[2]
)

ids = []
for i in relevant_sermons:
    ids.append(f"{i[1]} -- {i[0]}")

option = st.selectbox(
    "Welche Predigt soll analysiert werden (ID)?",
    ids
)

sermon = Sermon(option[-7:])

#############################
##### MAP VISUALISATION #####
#############################

def parse_coords(coord_str):
    try:
        if not coord_str:
            return None
        parts = coord_str.split(';')
        if len(parts) != 2:
            return None
        lon_str = parts[0].strip()
        lat_str = parts[1].strip()
        if lon_str[0] != 'E' or lat_str[0] != 'N':
            return None
        lon = float(lon_str[1:])
        lat = float(lat_str[1:])
        return lat, lon
    except:
        return None
    
author_network = sermon.autor.get_personal_network()

sermon_locations = {
    f"{sermon.einweihungsort.name} (Einweihungsort)": sermon.einweihungsort.koordinaten,
    f"{sermon.verlagsort.name} (Verlagsort)": sermon.verlagsort.koordinaten
}

# create folium Map
map = folium.Map(location=[50.8, 8.7], zoom_start=4)

# Add markers
for place, coord_str in author_network.items():
    coords = parse_coords(coord_str)
    if coords:
        folium.Marker(location=coords, popup=place, icon=folium.Icon(color='blue', icon='glyphicon-user')).add_to(map)

for place, coord_str in sermon_locations.items():
    coords = parse_coords(coord_str)
    if coords:
        folium.Marker(location=coords, popup=place, icon=folium.Icon(color='red', icon='glyphicon-book')).add_to(map)

###########################
##### QUOTATION PLOTS #####
###########################

color_map = {
    'bibel': 'tomato',
    'nan': 'steelblue',
    'quelle': 'limegreen',
    'literatur': 'teal',
    'musikwerk': 'lawngreen',
    'orgelpredigt': 'indigo'
}

##### quotation share pie chart
occurrences = {i:sermon.word_types.count(i) for i in set(sermon.word_types)}

labels = []
data = []

for label, number in occurrences.items():
    if pd.isnull(label):
        labels.append("text")
    else:
        labels.append(label.strip())
    data.append(number)

colors = [color_map.get(label, 'gray') for label in labels]

text_types_piechart = px.pie(values=data, names=labels, title='Anteile der Zitate am Gesamttext')

##### list of quotations
literaturliste = ""
lit_labels = []
lit_data = []
for quelle in sermon.literaturzitate:
    literaturliste += f"**Werk:** {str(quelle["item"])} **Anteil am Gesamttext:** {quelle["word_share"]/len(sermon.words)}\n\n"
    lit_labels.append(str(quelle["item"]))
    lit_data.append(quelle["word_share"])
for predigt in sermon.orgelpredigtzitate:
    literaturliste += f"**Werk:** {str(predigt["item"])} **Anteil am Gesamttext:** {predigt["word_share"]/len(sermon.words)}\n\n"
    lit_labels.append(str(predigt["item"]))
    lit_data.append(predigt["word_share"])

quotations_piechart = px.pie(values=lit_data, names=lit_labels, title='Verwendete Zitate')

quotations_piechart.update_layout(
    width=700,
    height=700,
    margin=dict(t=80, b=50, l=50, r=50),
    title_x=0.5,  # Center title
    legend=dict(
        orientation="h",  # horizontal legend
        y=-0.1  # push legend below chart
    )
)

##### quotation distribution over sermon in 100-Word-Chunks
overhang = len(sermon.words) % 100 
chunked_types=[]
for i in range(0,len(sermon.words),100):
    types = sermon.word_types[i:i+100]
    chunked_types.append(types)

last_types = sermon.word_types[-overhang:]
chunked_types.append(last_types)

quote_distribution_chunked = go.Figure(layout=dict(barmode='stack'))

x = [{"text": 95, "bible": 5}, {"text": 85, "bible": 6, "quelle": 9}, {"text": 100}, {"text": 77, "quelle": 10, "bibel": 3, "musikwerk": 10}]
for row, nr in zip(chunked_types, range(1, len(chunked_types))):
    item = dict(Counter(row))
    bar_title = f"Wörter 1 bis 100" if nr == 1 else f"Wörter {nr * 100} bis {(nr * 100) +100}"
    for key, val in item.items():
        quote_distribution_chunked.add_trace(go.Bar(
            name=str(key).strip(), 
            x=[bar_title], 
            y=[val],
            marker_color=color_map.get(str(key).strip(), 'gray')
            ))

quote_distribution_chunked.update_layout(barmode='stack')

##########################
##### STREAMLIT PAGE #####
##########################

st.set_page_config(
    page_title="Orgelpredigt_Analyse",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

st.title(f"{str(sermon)} – Analyse")

with st.sidebar:
    st.header("Information zur Predigt")
    st.markdown(f"**Predigtautor:** {sermon.autor.name}")
    st.markdown(f"**Titel:** {sermon.volltitel}")
    st.markdown(f"**Einweihungstag:** {sermon.sonntag}")
    st.markdown(f"**Einweihungsort:** {sermon.einweihungsort}")
    st.markdown(f"**Konfession:** {sermon.konfession}")
    st.markdown(f"**Bibelstelle:** {sermon.bibelstelle}")
    st.markdown(f"**Verleger:** {sermon.verleger.name}")
    st.markdown(f"**Verlagsort:** {sermon.verlagsort}")
    st.markdown(f"**Erscheinungsjahr:** {sermon.erscheinungsjahr}")
    st.markdown(f"**Umfang:** {sermon.umfang}")

    st.header("Information zum Autor")
    st.markdown(f"**Name:** {sermon.autor.name}")
    st.markdown(f"**Akademischer Grad:** {sermon.autor.akademisch}")
    st.markdown(f"**Geboren:** {sermon.autor.geburtsdatum} ({sermon.autor.geburtsort})")
    st.markdown(f"**Gestorben:** {sermon.autor.sterbedatum} ({sermon.autor.sterbeort})")
    st.markdown(f"**Funktionen:** {sermon.autor.funktionen}")


##### Geographischer Überblick

st.header("Geographischer Überblick zu Predigt und Biographie des Autors")
st.components.v1.html(folium.Figure().add_child(map).render(), height=500)


##### Überblick Zitate

st.header("Zitate")
st.plotly_chart(text_types_piechart)
st.plotly_chart(quotations_piechart)
st.markdown(literaturliste)

st.header("Verteilung von Zitaten im Text")
st.plotly_chart(quote_distribution_chunked)
