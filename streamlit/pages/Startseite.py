import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from core.utils import Sermon, Person, get_short_info
from collections import Counter

#import core.db_connection as db_connection

import plotly.express as px
from plotly.subplots import make_subplots
#px.colors.sequential.Agsunset
import networkx as nx

import plotly.graph_objects as go
import pandas as pd

import folium
import json
import re



from pathlib import Path

# root directory path
ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(
    page_title="Orgelpredigt-Zitate",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="auto",
    menu_items=None,
)

st.title('‘Das Wort sie sollen lassen stahn’ – Zitatnetzwerke in deutschsprachigen Orgelpredigten der frühen Neuzeit')
st.write('Click on the links below to navigate to the other pages.')

col1, col2 = st.columns([0.8,0.2], gap="small", vertical_alignment="top", border=False)

with col1:
    ##### Geographischer Überblick
    st.header("Predigt – Bibel – Kirchenlied: Eine Annäherung")
    st.write("Diese Seite bietet einen interaktiven Einstieg in die textuellen Netzwerke zwischen Lied- und Bibelzitat, in welchen sich frühmoderne Orgelpredigten bewegen. Die Startseite vermittelt hier einen generellen Überblick. Über 'Orgelpredigt Analyse' können einzelne Predigten genauer analysiert werden. Über 'Orgelpredigt Vergleich' können zwei oder mehr Predigten gemeinsam auf Überschneidungen hin untersucht werden.")

with col2:
    st.image(ROOT / "streamlit/mittweidische_orgel.jpg", caption="Prospekt der Weller-Orgel in Mittweida. Quelle: https://digital.slub-dresden.de/werkansicht?tx_dlf%5Bid%5D=10528&tx_dlf%5Bpage%5D=6#")

def create_legend(color_map):
    legend_translation = {
        "E00": "Predigt",
        "E10": "Musikwerk",
        "E08": "Quelle",
        "E09": "Literatur"
    }
    legend_traces = []

    for group_name, color in color_map.items():
        if group_name.startswith("E"):
            legend_traces.append(
                go.Scatter(
                    x=[None], y=[None],  # invisible point
                    mode='markers',
                    marker=dict(size=10, color=color),
                    legendgroup=group_name,
                    showlegend=True,
                    name=legend_translation[group_name]
                )
            )
    return legend_traces

#########################
##### CHOOSE SERMON #####
#########################

color_map = {
    'orgelpredigt': 'rgb(135, 44, 162)',
    'musikwerk': 'rgb(192, 54, 157)',
    'literatur': 'rgb(234, 79, 136)',
    'quelle': 'rgb(250, 120, 118)',
    'bibel': 'rgb(246, 169, 122)',
    'nan': 'rgb(237, 217, 163)',
    'text': 'rgb(237, 217, 163)',
    'E00': 'rgb(135, 44, 162)',
    'E10': 'rgb(192, 54, 157)',
    'E09': 'rgb(234, 79, 136)',
    'E08': 'rgb(250, 120, 118)'
    }

def is_id(value):
    pattern = re.compile(r'E[01][0-9]{5}')
    if re.match(pattern, value):
        return True
    else:
        return False

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

#########################
##### NETWORK GRAPH #####
#########################

sermons = []
for id in ids:
    item = {}
    current_sermon = Sermon(id)
    item["id"] = current_sermon.id
    item["links"] = [item for item in current_sermon.all_references if is_id(item)]
    sermons.append(item)

##### Sermons and Sources
G2 = nx.DiGraph()
nodes = []
connections = []
for sermon in sermons:
    nodes.append(sermon['id'])
    for link in sermon['links']:
        connections.append((sermon['id'], link))

G2.add_nodes_from(nodes)
G2.add_edges_from(connections)

in_degrees = dict(G2.in_degree())

pos = nx.spring_layout(G2, k=2, iterations=100)
degrees = dict(G2.degree())

for node in G2.nodes:
    G2.nodes[node]['pos'] = pos[node]
    assert 'pos' in G2.nodes[node], f"Node {node} missing 'pos'"
    assert G2.nodes[node]['pos'] is not None, f"Node {node} has None position"

mapping = {i: name for i, name in enumerate(ids)}
G2 = nx.relabel_nodes(G2, mapping)

edge_x = []
edge_y = []
edge_shapes = []
for edge in G2.edges():
    x0, y0 = G2.nodes[edge[0]]['pos']
    x1, y1 = G2.nodes[edge[1]]['pos']
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)
    
edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

node_x = []
node_y = []
node_sizes = []
node_colors = []
for node in G2.nodes():
    x, y = G2.nodes[node]['pos']
    node_x.append(x)
    node_y.append(y)
    node_sizes.append(degrees[node] * 10)
    node_colors.append(color_map.get(node[:3], 'gray'))

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    text=[n for n in G2.nodes()],
    marker=dict(
        showscale=False,
        size=node_sizes,
        colorscale='Magma',
        reversescale=False,
        color=node_colors,
        line_width=2))

in_degrees_list = [in_degrees[node] for node in G2.nodes]

node_adjacencies = []
node_text = []
in_connections = []
for node, adjacencies in enumerate(G2.adjacency()):
    node_adjacencies.append(len(adjacencies[1]))
    #node_text.append('# of connections: '+str(len(adjacencies[1])))
for node in G2.nodes:
    node_text.append(f"{get_short_info(node)} ({in_degrees[node]} Verweise)")
    in_connections.append(in_degrees[id])

node_trace.marker.size = [(x + 3) * 2.5  for x in in_degrees_list]
node_trace.text = node_text

sermons_sources_network = go.Figure(data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text="<br>Quotations among Sermons, Sources, and Music",
                    font=dict(size=16)
                    ),
                #shapes=edge_shapes,
                showlegend=True,
                hovermode='closest',
                margin=dict(b=40,l=10,r=10,t=80),
                annotations=[dict(
                    text="",
                    showarrow=True,
                    xref="paper", yref="paper",
                    x=0.00, y=-0.00 )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )

legend_traces = create_legend(color_map)

for trace in legend_traces:
    sermons_sources_network.add_trace(trace)

sermons_sources_network.update_layout(
    xaxis=dict(scaleanchor='y', scaleratio=1),
    yaxis=dict(scaleanchor='x', scaleratio=1),
    width=1200, height=1200,
    legend=dict(
        title='Kategorien',
        x=1.05,  # position legend to the right
        y=1,
        bgcolor='rgba(255,255,255,0.7)',
        bordercolor='black',
        borderwidth=1
    )
)

st.title("Literatur- und Liedzitate zwischen allen Orgelpredigten")
st.markdown("Der folgende Netzwerk-Graph visualisiert die Verweise in Predigttexten auf Literatur, Musikwerke, sowie andere Predigten.")
st.plotly_chart(sermons_sources_network)
st.title("Liedzitate – kumulativ und diachron betrachtet")

col1, col2 = st.columns([0.5, 0.5])

with col1:
    quote_type = st.selectbox(
            label="Zitatart auswählen",
            options=["musikwerk", "quelle", "orgelpredigt"],
            placeholder="musikwerk"
        )
with col2:
    quote_time_dist = st.selectbox(
            label="Zeitliche Einteilung",
            options=["ganzer Zeitraum", "50-Jahr-Intervalle", "25-Jahr-Intervalle"],
            placeholder="ganzer Zeitraum"
    )

def create_quote_dist_chart(ids: list, type: str) -> go.Figure:
    type_dict = {
        "orgelpredigt": "Orgelpredigtzitate",
        "musikwerk": "Liedzitate",
        "quelle": "Literaturzitate",
    }
    if type not in type_dict.keys():
        occ_fig = go.Figure()
        occ_fig.update_layout(title_text="Type not recognised!")
        return occ_fig
    
    else:
        chunked_text = [0]*100
        thumbnails = [""]*100

        for id in ids:
            sermon = Sermon(id)

            dec = int(len(sermon.words) / 99)
            overhang = len(sermon.words) % dec

            for i, j in zip(range(0, len(sermon.words), dec), range(0, 100)):
                types_unique = list(set(sermon.word_types[i:i+dec]))
                types_str = " ".join([x for x in types_unique if isinstance(x, str)])
                if type in types_str:
                    type_test = 1
                    hit = f"{sermon.kurztitel}<br>"
                else:
                    type_test = 0
                    hit = ""
                
                chunked_text[j] = chunked_text[j] + type_test
                thumbnails[j] = thumbnails[j] + hit
            
            last_types_unique = list(set(sermon.word_types[-overhang:]))
            last_types_str = " ".join([x for x in last_types_unique if isinstance(x, str)])
            if type in last_types_str:
                last_type_test = 1
                last_hit = f"{sermon.kurztitel}<br>"
            else:
                last_type_test = 0
                last_hit = ""
            
            #chunked_text[-1] = chunked_text[-1] + last_orgelpredigt_test
            #thumbnails[-1] = thumbnails[-1] + last_hit

        occ_fig = go.Figure()

        for i in range(0, len(chunked_text)):
            hovertext = f'{chunked_text[i]} {type_dict[type]} im {i+1}%'
            if thumbnails[i] != "":
                    hovertext += f"<br>{thumbnails[i]}"

            gradient = chunked_text[i] * 15
            color = f'rgb({max(250-gradient, 0)},{max(250-gradient, 0)},{max(250-gradient, 0)})'
            occ_fig.add_trace(go.Bar(
                x = [f"{type_dict[type]} je Predigtprozent"],
                y = [100],
                marker_color = color,
                hovertext = hovertext
            ))

        occ_fig.update_layout(width=1500,height=500, showlegend=False)
            
        return occ_fig

def group_sermons_in_years(data, interval: int) -> list:
    chunked_sermons = []
    start_year = 1600
    end_year = 1800
    yearfinder = re.compile(r'[0-9]{4}')
    for i in range(start_year, end_year, interval):
        sermons = []
        for id, info in data.items():
            year = int(re.findall(yearfinder, info['year'])[0])
            if year > i and year < i + interval:
                sermons.append(id)
        chunked_sermons.append(sermons)

    return chunked_sermons

if quote_time_dist == "50-Jahr-Intervalle":
    sermons_grouped_50 = group_sermons_in_years(data, 50)
    figs_50 = []
    for i in range(len(sermons_grouped_50)):
        figs_50.append(create_quote_dist_chart(sermons_grouped_50[i], quote_type))
    
    # Create subplots
    fig = make_subplots(rows=len(figs_50), 
                        cols=1, 
                        subplot_titles=[f"Verteilung in Predigten zwischen {1600 + (i*50)} und {1600+(i*50)+50} ({len(sermons_grouped_50[i])} Predigten)" for i in range(len(figs_50))])

    # Add traces from each figure to the subplots
    for i, fig_item in enumerate(figs_50):
        for trace in fig_item.data:
            fig.add_trace(trace, row=i+1, col=1)

    # Update layout
    fig.update_layout(height=1200, width=1000, showlegend = False)
    fig.update_layout(title_text="Accumulierte Verteilung von Zitaten in 50-Jahr Intervallen")

elif quote_time_dist == "25-Jahr-Intervalle":
    sermons_grouped_25 = group_sermons_in_years(data, 25)
    figs_25 = []
    for i in range(len(sermons_grouped_25)):
        figs_25.append(create_quote_dist_chart(sermons_grouped_25[i], quote_type))
    
    # Create subplots
    fig = make_subplots(rows=len(figs_25), 
                        cols=1, 
                        subplot_titles=[f"Verteilung in Predigten zwischen {1600 + (i*25)} und {1600+(i*25)+25} ({len(sermons_grouped_25[i])} Predigten)" for i in range(len(figs_25))])

    # Add traces from each figure to the subplots
    for i, fig_item in enumerate(figs_25):
        for trace in fig_item.data:
            fig.add_trace(trace, row=i+1, col=1)

    # Update layout
    fig.update_layout(height=1200, width=1000, showlegend = False)
    fig.update_layout(title_text="Accumulierte Verteilung von Zitaten in 25-Jahr Intervallen")

else:
    fig = create_quote_dist_chart(ids, quote_type)

st.plotly_chart(fig)