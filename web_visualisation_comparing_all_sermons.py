import streamlit as st
from orgelpredigt_analysis import Sermon
from collections import Counter

import plotly.express as px
#px.colors.sequential.Agsunset

import plotly.graph_objects as go
import pandas as pd

import folium
import json
import re
import os

color_map = {
    'orgelpredigt': 'rgb(135, 44, 162)',
    'musikwerk': 'rgb(192, 54, 157)',
    'literatur': 'rgb(234, 79, 136)',
    'quelle': 'rgb(250, 120, 118)',
    'bibel': 'rgb(246, 169, 122)',
    'nan': 'rgb(237, 217, 163)',
    'text': 'rgb(237, 217, 163)'
    }

def is_id(value):
    pattern = re.compile(r'E[01][0-9]{5}')
    if re.match(pattern, value):
        return True
    else:
        return False

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

ids = [i[0] for i in relevant_sermons]



def create_stacked_chart(sermon):

    overhang = len(sermon.words) % 100 
    chunked_types=[]
    for i in range(0,len(sermon.words),100):
        types = ["text" if isinstance(x, float) else x for x in sermon.word_types[i:i+100]]
        reference = [" ".join(ref) for ref in sermon.reference[i:i+100]]
        concat = [",".join(zipped) for zipped in list(zip(types, reference))]
        chunked_types.append(concat)

    last_types = ["" if isinstance(x, float) else x for x in sermon.word_types[-overhang:]]
    last_refs = [" ".join(ref) for ref in sermon.reference[-overhang:]]
    last_concat = [",".join(zipped) for zipped in list(zip(last_types, last_refs))]
    chunked_types.append(last_concat)

    quote_distribution_chunked = go.Figure(layout=dict(barmode='stack'))

    for row, nr in zip(chunked_types, range(1, len(chunked_types))):
        item = dict(Counter(row))
        bar_title = f"Wörter 1 bis 100" if nr == 1 else f"Wörter {nr * 100} bis {(nr * 100) +100}"
        for key, val in item.items():
            color, ref = key.split(',')
            name = f'{str(ref).strip()}' if is_id(ref) else str(key).replace(',', '').strip()
            url=f'https://orgelpredigt.ur.de/{str(ref).strip()}' if is_id(ref) else ""
            quote_distribution_chunked.add_trace(go.Bar(
                name=name, 
                x=[bar_title], 
                y=[val],
                hovertemplate=f'<b>{name}</b><br>Value: {val} Words<br>Link: <a href="{url}">{ref}</a><extra></extra>',
                marker_color=color_map.get(str(color).strip(), 'gray')
                ))

    quote_distribution_chunked.update_layout(barmode='stack')
    
    return quote_distribution_chunked

##########################
##### STREAMLIT PAGE #####
##########################

st.set_page_config(
    page_title="Orgelpredigt_Analyse",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None)

st.title("Vergleich aller Orgelpredigten")

for id in ids:
    print(id)
    sermon = Sermon(id)
    st.markdown(sermon.kurztitel)
    st.plotly_chart(create_stacked_chart(sermon))
    st.divider()

         