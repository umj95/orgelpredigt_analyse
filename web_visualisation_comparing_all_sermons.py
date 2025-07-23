import streamlit as st
from orgelpredigt_analysis import Sermon, Person
from collections import Counter

import db_connection

import plotly.express as px
#px.colors.sequential.Agsunset
import networkx as nx

import plotly.graph_objects as go
import pandas as pd

import folium
import json
import re
import os

cursor, connection = db_connection.get_connection()

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

def get_short_info(id):
    cursor = connection.cursor()
    if is_id(id):
        if id.startswith("E08"):
            try:
                cursor.execute(f"SELECT e08autor1, e08titel1, e08ort, e08jahr FROM e08_quellen WHERE e08id = '{id}'")
                column_names = [col[0] for col in cursor.description]
                results = cursor.fetchall()
                if results:
                    data = [dict(zip(column_names, row)) for row in results][0]
                    author = data.get("e08autor1", "no author")
                    title = data.get("e08titel1", "no title")
                    place = data.get("e08ort", "s.l.")
                    year = data.get("e08jahr", "s.a.")
                    return f"{author}: {title} ({place}, {year})"
                else:
                    return "no data for this source"
            except Error as e:
                print(f"Database error occurred for {id}:", e)
            except Exception as e:
                print(f"Unexpected error for {id}:", e)
        elif id.startswith("E09"):
            try:
                cursor.execute(f"SELECT e09autor1, e09titel1, e09ort, e09jahr FROM e09_literatur WHERE e09id = '{id}'")
                column_names = [col[0] for col in cursor.description]
                results = cursor.fetchall()
                if results:
                    data = [dict(zip(column_names, row)) for row in results][0]
                    author = data.get("e09autor1", "no author")
                    title = data.get("e09titel1", "no title")
                    place = data.get("e09ort", "s.l.")
                    year = data.get("e09jahr", "s.a.")
                    return f"{author}: {title} ({place}, {year})"
                else:
                    return "no data for this source"
            except Error as e:
                print(f"Database error occurred for {id}:", e)
            except Exception as e:
                print(f"Unexpected error for {id}:", e)
        elif id.startswith("E10"):
            try:
                cursor.execute(f"SELECT e10komponist, e10werk FROM e10_musikwerke WHERE e10id = '{id}'")
                column_names = [col[0] for col in cursor.description]
                results = cursor.fetchall()
                if results:
                    data = [dict(zip(column_names, row)) for row in results][0]
                    composer = data.get("e10komponist", "no composer")
                    title = data.get("e10werk", "no title")
                    return f"{composer}: {title}"
                else:
                    return "no data for this source"
            except Error as e:
                print(f"Database error occurred for {id}:", e)
            except Exception as e:
                print(f"Unexpected error for {id}:", e)
        elif id.startswith("E00"):
            try:
                cursor.execute(f"SELECT e00autor, e00kurztitel FROM e00_orgelpredigten WHERE e00id = '{id}'")
                column_names = [col[0] for col in cursor.description]
                results = cursor.fetchall()
                if results:
                    sermon_info = [dict(zip(column_names, row)) for row in results][0]
                    author = Person(sermon_info["e00autor"]).name
                    title = sermon_info["e00kurztitel"]
                    return f"{author}: {title}"
                else:
                    return "no data for this source"
            except Error as e:
                print(f"Database error occurred for {id}:", e)
            except Exception as e:
                print(f"Unexpected error for {id}:", e)
        else:
            return f"{id}"

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

##### Only Sermons
G = nx.DiGraph()

nodes = []
connections = []
for sermon in sermons:
    nodes.append(sermon['id'])
    for link in sermon['links']:
        if re.match(r'E00[0-9]{4}', link):
            connections.append((sermon['id'], link))

G.add_nodes_from(nodes)
G.add_edges_from(connections)

in_degrees = dict(G.in_degree()) # compute incoming connections for each node
degrees = dict(G.degree())

pos = nx.kamada_kawai_layout(G)
for node in G.nodes:
    G.nodes[node]['pos'] = pos[node]
    assert 'pos' in G.nodes[node], f"Node {node} missing 'pos'"
    assert G.nodes[node]['pos'] is not None, f"Node {node} has None position"

mapping = {i: name for i, name in enumerate(ids)}
G = nx.relabel_nodes(G, mapping)

edge_x = []
edge_y = []
edge_shapes = []
for edge in G.edges():
    x0, y0 = G.nodes[edge[0]]['pos']
    x1, y1 = G.nodes[edge[1]]['pos']
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
for node in G.nodes():
    x, y = G.nodes[node]['pos']
    node_x.append(x)
    node_y.append(y)
    node_sizes.append(degrees[node] * 10)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    text=[n for n in G.nodes()],
    marker=dict(
        showscale=True,
        size=node_sizes,
        colorscale='Magma',
        reversescale=False,
        color=[],
        colorbar=dict(
            thickness=15,
            title=dict(
              text='Node Connections',
              side='right'
            ),
            xanchor='left',
        ),
        line_width=2))

node_adjacencies = []
node_text = []
in_connections = []
for node, adjacencies in enumerate(G.adjacency()):
    node_adjacencies.append(len(adjacencies[1]))
    #node_text.append('# of connections: '+str(len(adjacencies[1])))
for id in ids:
    node_text.append(f"{get_short_info(id)} ({in_degrees[id]} Verweise)")
    in_connections.append(in_degrees[id])

node_trace.marker.color = in_connections
node_trace.marker.size = [(x + 4) * 3  for x in in_connections]
node_trace.text = node_text

sermons_network_graph = go.Figure(data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text="<br>Quotations in between sermons",
                    font=dict(size=16)
                    ),
                #shapes=edge_shapes,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=40,l=10,r=10,t=80),
                width=600, height=600,
                annotations=[dict(
                    text="",
                    showarrow=True,
                    xref="paper", yref="paper",
                    x=0.00, y=-0.00 )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )

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

##########################
##### STACKED CHARTS #####
##########################

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
            colors, ref = key.split(',')
            colors = colors.strip().split(" ")
            if len(colors) > 1:
                color = colors[1] if is_id(colors[1]) else colors[0]
            else:
                color = colors[0]
            key_cleaned = str(key).replace(',', '').strip().split('.')[0]
            name = f'{" ".join(get_short_info(str(ref)).split()[:6])}' if is_id(ref) else key_cleaned
            url=f'https://orgelpredigt.ur.de/{str(ref).strip()}' if is_id(ref) else ""
            quote_distribution_chunked.add_trace(go.Bar(
                name=name, 
                x=[bar_title], 
                y=[val],
                hovertemplate=f'<b>{name}</b><br>Value: {val} Words<br>Link: <a href="{url}">{ref}</a><extra></extra>',
                marker_color=color_map.get(str(color).strip(), 'gray')
                ))

    quote_distribution_chunked.update_layout(barmode='stack',
                                             width=800,
                                             height=300)
    
    return quote_distribution_chunked

##########################
##### STREAMLIT PAGE #####
##########################

st.set_page_config(
    page_title="Orgelpredigt_Analyse",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="auto",
    menu_items=None)

st.title("Vergleich aller Orgelpredigten")
st.plotly_chart(sermons_network_graph)
st.plotly_chart(sermons_sources_network)

for id in ids:
    print(id)
    sermon = Sermon(id)
    st.markdown(f"**{sermon.kurztitel}**")
    st.plotly_chart(create_stacked_chart(sermon))
    #st.divider()

         