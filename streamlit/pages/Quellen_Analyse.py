import sys
import os

# Get the absolute path to the repository root
from pathlib import Path

# root directory path
#root = Path(os.getcwd()).resolve().parents[1]

# Add the repository root to the Python path
#sys.path.append(str(root))

#repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#if repo_root not in sys.path:
#    sys.path.insert(0, repo_root)

import streamlit as st
from core.utils import Sermon, get_short_info

from collections import Counter
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from plotly.subplots import make_subplots
import pandas as pd

import folium
import json
import re
from pathlib import Path

from rapidfuzz import fuzz

# root directory path
ROOT = Path(__file__).resolve().parents[2]

print("WORKING DIR:", os.getcwd())
print("SYS.PATH:", sys.path)

st.set_page_config(
    page_title="Orgelpredigt-Analyse",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="auto",
    menu_items=None,
)

color_map = {
        'orgelpredigt': 'rgb(135, 44, 162)',
        'musikwerk': 'rgb(192, 54, 157)',
        'literatur': 'rgb(234, 79, 136)',
        'quelle': 'rgb(250, 120, 118)',
        'bibel': 'rgb(246, 169, 122)',
        'machine_orgelpredigt': 'rgb(135, 44, 162)',
        'machine_musikwerk': 'rgb(192, 54, 157)',
        'machine_literatur': 'rgb(234, 79, 136)',
        'machine_quelle': 'rgb(250, 120, 118)',
        'machine_bibel': 'rgb(246, 169, 122)',
        'nan': 'rgb(237, 217, 163)',
        'text': 'rgb(237, 217, 163)'
    }

def is_id(value):
    pattern = re.compile(r'E[01][0-9]{5}')
    if re.match(pattern, value):
        return True
    else:
        return False
    
def flatten(xss):
    return [x for xs in xss for x in xs]
def create_quote_dist_chart(ids: list, type: str, searchkey: str="") -> go.Figure:
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
        colors_cumulative = {
            'orgelpredigt': (219, 192, 227),
            'musikwerk': (236, 195, 226),
            'quelle': (255, 196, 197)
        }
        red = colors_cumulative[type][0]
        green = colors_cumulative[type][1]
        blue = colors_cumulative[type][2]

        chunked_text = [0]*100
        thumbnails = [""]*100

        for id in ids:
            sermon = Sermon(id)

            dec = int(len(sermon.words) / 99)
            overhang = len(sermon.words) % dec

            for i, j in zip(range(0, len(sermon.words), dec), range(0, 100)):
                if searchkey != "":
                    keys_unique = sermon.reference[i:i+dec]
                    keys_str = " ".join(flatten(keys_unique))
                    if searchkey in keys_str:
                        hit_test = 1
                        hit = f"{sermon.kurztitel}"
                    else:
                        hit_test = 0
                        hit = ""
                else:
                    types_unique = list(set(sermon.word_types[i:i+dec]))
                    types_str = " ".join([x for x in types_unique if isinstance(x, str)])
                    if type in types_str:
                        hit_test = 1
                        hit = f"{sermon.kurztitel}"
                    else:
                        hit_test = 0
                        hit = ""
                
                chunked_text[j] = chunked_text[j] + hit_test
                thumbnails[j] = thumbnails[j] + hit
            if searchkey != "":
                last_keys_unique = sermon.reference[-overhang:]
                last_keys_str = " ".join(flatten(last_keys_unique))
                if searchkey in last_keys_str:
                    last_hit_test = 1
                    last_hit = f"{sermon.kurztitel}"
                else:
                    last_hit_test = 0
                    last_hit = ""
            else:
                last_types_unique = list(set(sermon.word_types[-overhang:]))
                last_types_str = " ".join([x for x in last_types_unique if isinstance(x, str)])
                if type in last_types_str:
                    last_hit_test = 1
                    last_hit = f"{sermon.kurztitel}"
                else:
                    last_hit_test = 0
                    last_hit = ""
            
            #chunked_text[-1] = chunked_text[-1] + last_orgelpredigt_test
            #thumbnails[-1] = thumbnails[-1] + last_hit

        occ_fig = go.Figure()

        for i in range(0, len(chunked_text)):
            hovertext = f'{chunked_text[i]} {type_dict[type]} im {i+1}%'
            if thumbnails[i] != "":
                    hovertext += f"{thumbnails[i]}"

            gradient = chunked_text[i] * 15
            #color = f'rgb({max(250-gradient, 0)},{max(250-gradient, 0)},{max(250-gradient, 0)})'
            if gradient > 0:
                color = f'rgb({red},{max(green-gradient, 0)},{max(250-gradient, 0)})'
            else:
                color = 'rgb(245,245,245)'
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

def get_closest_verse(sermon_line: str, source: list) -> list:
    best_match = ""
    best_score = 0
    for verse in source:
        sim_score = fuzz.ratio(sermon_line, verse.lower())
        if sim_score > best_score:
            best_score = sim_score
            best_match = verse
    
    return [best_match, best_score, source.index(best_match)]

def get_quoted_passages(quoted_item: str, ids: list):
    # get passages from quoted item in every sermon
    passages = []
    for id in ids:
        sermon = Sermon(id)
        for i in range(len(sermon.chunked)):
            for j in range(len(sermon.chunked[i])):
                words = sermon.chunked[i][j]["words"]
                types = sermon.chunked[i][j]["types"]
                refs = sermon.chunked[i][j]["references"]
                test_refs = []
                for nr, ref in enumerate(refs):
                    if quoted_item in ref:
                        test_refs.append(nr)
                if len(test_refs):
                    if j > 0:
                        words_before = " ".join(sermon.chunked[i][j-1]["words"])
                    else:
                        words_before = " ".join(sermon.chunked[i-1][-1]["words"])
                    if j < len(sermon.chunked[i])-1:
                        words_after = " ".join(sermon.chunked[i][j+1]["words"])
                    elif i + 1 < len(sermon.chunked):
                        words_after = " ".join(sermon.chunked[i+1][0]["words"])
                    else:
                        words_after = ""
                    passages.append([id, " ".join(words), words_before, words_after])
    
    return passages

def style_quote(verse_dict: dict) -> str:
    color_map = {'lieder':[192, 54, 157]}
    opacity = (verse_dict["quotes"] * 10) / 100
    bg_color = color_map["lieder"]
    bg_color.append(opacity)  # type: ignore
    color_str = ", ".join([str(i) for i in bg_color])
    attr = f'style="background-color: rgba({color_str}); color: {"white" if opacity > 0.3 else "#5B5B66"}; border-radius: 5px; padding: 2px;"'
    return attr

def create_tooltip(verse_dict: dict) -> str:
    tooltip = ""
    if verse_dict["quotes"] > 0:
        tooltip = """<div class="tooltip-content">"""
        for item in verse_dict["details"]:
            if is_id(item[0]):
                sermon = Sermon(item[0])
                title_info = f'{sermon.autor.name}<br/><h4><a href="https://orgelpredigt.ur.de/{item[0]}" target="_blank">{sermon.kurztitel}</a></h4>'
            else:
                title_info = item[0]
            text = item[1]
            text_before = item[2]
            text_after = item[3]
            text_context = f'{text_before} <span class="actual_quote">{text}</span> {text_after}'
            score = f"{item[5]:.4f}"
            div = f"""<div class="info-box">
                            {title_info}
                            <p>{text_context}</p>
                            <div class="score">Ähnlichkeit: {score}%</div>
                        </div>"""
            tooltip += div
        tooltip += "</div>"
    return tooltip

st.markdown("Welches Lied soll analysiert werden? Bitte aus dem Dropdown-Menü auswählen")
sources = ["In dulci jubilo", "Wie schön leuchtet der Morgenstern", "Ach Gott, wie manches Herzeleid"]
option = st.selectbox(
    label="Lied auswählen",
    options=sources,
    placeholder="Lied")

id_map = {"In dulci jubilo": "E100017", "Wie schön leuchtet der Morgenstern": "E100022", "Ach Gott, wie manches Herzeleid": "E100033"}

selected_id = id_map[option]
print(selected_id)

# Get the list of all files in a directory
with open(ROOT / "predigten_übersicht.json", "r", encoding="utf-8") as file: 
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

ids = [x[0] for x in relevant_sermons]

# get the song texts and select the right one
with open(ROOT / "source_texts/songs.json", "r", encoding="utf-8") as f:
    songtexts = json.load(f)

songtext = songtexts[selected_id]

passages = get_quoted_passages(selected_id, ids)
results = pd.DataFrame(passages, columns=['sermon', 'verse', 'text_before', 'text_after'])

results[["best_match", "sim_score", "line"]] = results["verse"].apply(lambda x: pd.Series(get_closest_verse(x, songtext)))
lines_with_data = []
for line in songtext:
    row_list = results.loc[results["best_match"] == line].values.tolist()
    lines_with_data.append({"name": line, "value":row_list})

lines_cleaned = []
for line in lines_with_data:
    line_info = {}
    line_info["text"] = line["name"]
    line_info["quotes"] = len(line["value"])
    line_info["details"] = line["value"]
    lines_cleaned.append(line_info)

text = '<div class="song">'
for line in lines_cleaned:
    text += f'<span class="verse {"tooltip" if line["quotes"] > 0 else ""}" {style_quote(line)}>{line["text"]}{create_tooltip(line)}</span><br/>'
text += "</div>"

## More info about the sermons
quoting_sermons = set(results['sermon'].values.tolist())

def year_helper(year):
    year_finder = re.compile(r'[0-9]{4}')
    year_cleaned = re.findall(year_finder, year)[0]
    return year_cleaned

sermon_info = []
for elem in quoting_sermons:
    sermon = Sermon(elem)
    info = {'id': elem, 'author': sermon.autor.name,'title': sermon.kurztitel, 'year': year_helper(sermon.erscheinungsjahr), 'place': sermon.verlagsort, 'coords': sermon.einweihungsort.koordinaten}
    sermon_info.append(info)

st.markdown("""
    <style>
        /* Tooltip container */
        .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
        }

        /* Tooltip content */
        .tooltip-content {
        display: flex;
        gap: 0.5rem;
        position: absolute;
        top: 120%;
        left: 0; /* default: align left */
        background: #f9f9f9;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.25s ease-in-out;
        max-width: calc(100vw - 20px); /* never overflow window */
        overflow-x: auto; /* allow scrolling if too wide */
        white-space: nowrap;
        z-index: 1000;
        }

        /* Tooltip arrow */
        .tooltip-content::before {
        content: "";
        position: absolute;
        top: -8px;
        left: 20px; /* default arrow position */
        border-width: 8px;
        border-style: solid;
        border-color: transparent transparent #f9f9f9 transparent;
        }

        /* Show tooltip on hover */
        .tooltip:hover .tooltip-content {
        opacity: 1;
        pointer-events: auto;
        }

        /* Individual info boxes */
        .info-box {
        flex: 0 0 auto;
        background: #fff;
        border: 1px solid #ddd;
        border-radius: 6px;
        padding: 0.5rem;
        width: 12rem;
        display: flex;
        flex-direction: column;
        justify-content: start  ;
        white-space: normal;
        color: #5B5B66;
        }

        /* Name */
        .info-box h4 {
        margin: 0;
        font-size: 0.95rem;
        font-weight: bold;
        color: #333;
        }

        /* Line of text */
        .info-box p {
        margin: 0.25rem 0;
        font-size: 0.85rem;
        color: #555;
        }

        /* Numeric score */
        .info-box .score {
        /*font-weight: bold;
        font-size: 1rem;
        color: #2a7a2a;*/
        text-align: right;
        margin-top: auto;
        }
        
        .info-box .actual_quote {
        color: rgb(192, 54, 157);
            }
    </style>
    """, unsafe_allow_html=True)
st.header(f"{option}")
col1, col2 = st.columns([0.3,0.7], gap="small", vertical_alignment="top", border=False)
with col1:
    st.html(text)
with col2:
    st.markdown(f"**Predigten, die „{option}“ zitieren**")
    for x in sermon_info:
        st.markdown(f"- {x["author"]}: [{x["title"]}](https://orgelpredigt.ur.de/{x["id"]})")
    st.header("Akkumulierte Verteilung der Zitate über alle Predigten")
    quote_time_dist = st.selectbox(
            label="Zeitliche Einteilung",
            options=["ganzer Zeitraum", "50-Jahr-Intervalle", "25-Jahr-Intervalle"],
            placeholder="ganzer Zeitraum"
    )

    quote_type = "musikwerk"
    searchkey= selected_id
    if quote_time_dist == "50-Jahr-Intervalle":
        sermons_grouped_50 = group_sermons_in_years(data, 50)
        figs_50 = []
        for i in range(len(sermons_grouped_50)):
            figs_50.append(create_quote_dist_chart(sermons_grouped_50[i], quote_type,searchkey=searchkey))
        
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
        fig.update_layout(title_text="Akkumulierte Verteilung von Zitaten in 50-Jahr Intervallen")

    elif quote_time_dist == "25-Jahr-Intervalle":
        sermons_grouped_25 = group_sermons_in_years(data, 25)
        figs_25 = []
        for i in range(len(sermons_grouped_25)):
            figs_25.append(create_quote_dist_chart(sermons_grouped_25[i], quote_type,searchkey=searchkey))
        
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
        fig.update_layout(title_text="Akkumulierte Verteilung von Zitaten in 25-Jahr Intervallen")

    else:
        fig = create_quote_dist_chart(ids, quote_type, searchkey=searchkey)
    st.plotly_chart(fig)

    years = [int(x["year"]) for x in sermon_info]
    # Create DataFrame
    df = pd.DataFrame(years, columns=["year"])

    # Filter between 1600 and 1800
    df = df[(df["year"] >= 1600) & (df["year"] < 1800)]

    # Create a decade column
    df["decade"] = (df["year"] // 10) * 10

    # Count publications per decade
    counts = df["decade"].value_counts().sort_index()

    # Reindex to include all decades even if 0
    all_decades = pd.Series(0, index=range(1600, 1800, 10))
    counts = all_decades.add(counts, fill_value=0).astype(int)

    # Convert to DataFrame for Plotly
    counts_df = counts.reset_index()
    counts_df.columns = ["decade", "count"]

    # Plot
    fig = px.bar(
        counts_df,
        x="decade",
        y="count",
        title="Publikationen pro Dekade (1600–1800)",
        labels={"decade": "Dekade", "count": "Anzahl der Publikationen"}
    )

    # Update bar color
    fig.update_traces(marker_color="rgb(135, 44, 162)")

    # Ensure y-axis ticks are integers only
    fig.update_layout(
        xaxis=dict(dtick=10),
        yaxis=dict(tickmode="linear", dtick=1))
    
    st.header("Zeitliche Verteilung")
    st.plotly_chart(fig)