import sys
import os

# Get the absolute path to the repository root
from pathlib import Path

# root directory path
root = Path(os.getcwd()).resolve().parents[1]

# Add the repository root to the Python path
sys.path.append(str(root))

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
# root directory path
ROOT = Path(__file__).resolve().parents[2]

import streamlit as st
from core.utils import Sermon, get_short_info

from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from plotly.subplots import make_subplots
import pandas as pd
import json
import re

from rapidfuzz import fuzz

#print("WORKING DIR:", os.getcwd())
#print("SYS.PATH:", sys.path)

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
    
def flatten(xss):
    return [x for xs in xss for x in xs]

def get_shared_quotes(quoted_item: str, ids: list):
    shared_quotes = []
    for id in ids:
        sermon = Sermon(id)
        for par in sermon.chunked:
            refs = flatten(flatten([x["references"] for x in par]))
            if quoted_item in refs:
                shared_quotes.extend(set(refs))
    return [x for x in shared_quotes if x != quoted_item]

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
                    passages.append([id, i, j, " ".join(words), words_before, words_after])
    
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
            text = item[3]
            text_before = item[4]
            text_after = item[5]
            text_context = f'{text_before} <span class="actual_quote">{text}</span> {text_after}'
            score = f"{item[7]:.4f}"
            div = f"""<div class="info-box">
                            {title_info}
                            <p>{text_context}</p>
                            <div class="score">Ähnlichkeit: {score}%</div>
                        </div>"""
            tooltip += div
        tooltip += "</div>"
    return tooltip

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
                    passages.append([id, i, j, " ".join(words), words_before, words_after])
    
    return passages

def get_full_quotes(df):
    # Ensure proper order
    df = df.sort_values(["sermon","par","sent"]).reset_index(drop=True)

    # Detect breaks in consecutiveness *per sermon and par*
    df["break"] = (
        (df["sent"] != df["sent"].shift(1) + 1) | 
        (df["par"] != df["par"].shift(1)) |
        (df["sermon"] != df["sermon"].shift(1))
    )

    # Create group id
    df["grp"] = df["break"].cumsum()

    # Group and concatenate
    # aggregate while keeping sermon/par in output
    out = (
        df.groupby(["sermon","par","grp"], as_index=False)
        .agg(
            sent_min = ("sent","min"),
            sent_max = ("sent","max"),
            verse = ("verse", " ".join)
        )
    )

    # optional: pretty sent_range and drop grp
    out["sent_range"] = out.apply(
        lambda r: str(r["sent_min"]) if r["sent_min"]==r["sent_max"] else f"{r['sent_min']}-{r['sent_max']}",
        axis=1
    )
    out = out.drop(columns="grp")
    return out

def get_sent_before(id:  str, par: int, sent: int) -> str:
    sermon = Sermon(id)
    if sent != 0:
        return " ".join(sermon.chunked[par][sent-1]["words"])
    else:
        return ""
def get_sent_after(id:  str, par: int, sent: int) -> str:
    sermon = Sermon(id)
    if sent+1 != len(sermon.chunked[par]):
        return " ".join(sermon.chunked[par][sent+1]["words"])
    else:
        return ""
    
def create_html(df):
    df["formatted"] = df.apply(
    lambda r: f'{r['before']} <span class="actual_quote">{r['verse']}</span>{r['after']}', axis=1
)
    sermon_dict = df.groupby("sermon")["formatted"].agg(list).to_dict()
    return sermon_dict

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

source_options = []
for id in ids:
    sermon = Sermon(id)
    source_options.extend(set(flatten(sermon.reference)))

source_options = [x for x in source_options if is_id(x)]

id_counts = Counter(source_options)
# Sort IDs by frequency descending
sorted_ids = [[item[0], item[1]] for item in id_counts.most_common()]
options = [{f'{get_short_info(i[0])} (Zitiert in {i[1]} Predigt{"en" if i[1]>1 else ""})' :i[0]} for i in sorted_ids]

keys = []
for x in options:
    key = [key for key, val in x.items()]
    keys.extend(key)

st.markdown("Was soll analysiert werden? Bitte aus dem Dropdown-Menü auswählen.")
st.markdown("Für die Lieder 'In dulci jubilo', 'Wie schön leuchtet der Morgenstern' und 'Ach Gott, wie manches Herzeleid', sowie für Conrad Dieterichs 'Ulmische Orgelpredigt' kann auch der Volltext mit hervorgehobenen Zitaten angezeigt werden")

sources = ["In dulci jubilo", "Wie schön leuchtet der Morgenstern", "Ach Gott, wie manches Herzeleid", "Ulmische Orgelpredigt"]
option = st.selectbox(
    label="Lied auswählen",
    options=keys,
    placeholder="Lied")

id_map = {"In dulci jubilo": "E100017", "Wie schön leuchtet der Morgenstern": "E100022", "Ach Gott, wie manches Herzeleid": "E100033", "Ulmische Orgelpredigt": "E000003"}

#selected_id = id_map[option]

selected_id = [x[option] for x in options if option in x.keys()][0]

print(selected_id)


# get the song texts and select the right one
with open(ROOT / "source_texts/songs.json", "r", encoding="utf-8") as f:
    songtexts = json.load(f)

songtext = False

if selected_id in ["E100017", "E100022", "E100033"]:
    songtext = songtexts[selected_id]
elif selected_id == "E000033":
    sermon_chunks = Sermon(selected_id).chunked
    sermon_verses = []
    for par in sermon_chunks:
        for sent in par:
            sermon_verses.append(" ".join(sent['words']))
    songtext = sermon_verses

passages = get_quoted_passages(selected_id, ids)
results = pd.DataFrame(passages, columns=['sermon', 'par', 'sent', 'verse', 'text_before', 'text_after'])

full_quotes = get_full_quotes(results)
full_quotes["before"] = full_quotes.apply(
    lambda r: get_sent_before(r["sermon"], r["par"], r["sent_min"]), axis=1
)
full_quotes["after"] = full_quotes.apply(
    lambda r: get_sent_after(r["sermon"], r["par"], r["sent_max"]), axis=1
)

quote_dict = create_html(full_quotes)

if songtext:
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
else:
    text = "<b>Der Text für diese Quelle liegt nicht vor</b>"

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
        
        .actual_quote {
        color: rgb(192, 54, 157);
            }
        
        .full_quote {
        background: #f9f9f9;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        opacity: 1;
        pointer-events: none;
        transition: opacity 0.25s ease-in-out;
        max-width: calc(100vw - 20px);
        overflow-x: auto;
        white-space: nowrap;
        z-index: 1000;
        }

        /* Individual info boxes */
        .quote-box {
        flex: 0 0 auto;
        background: #fff;
        border: 1px solid #ddd;
        border-radius: 6px;
        padding: 0.5rem;
        display: flex;
        flex-direction: column;
        justify-content: start  ;
        white-space: normal;
        color: #5B5B66;
        }
    </style>
    """, unsafe_allow_html=True)
st.header(f"{option}")
col1, col2 = st.columns([0.4,0.6], gap="small", vertical_alignment="top", border=False)
with col1:
    st.html(text)
with col2:
    with st.expander(f"Passagen die '{option}' zitieren anzeigen"):
        for x in sermon_info:
            st.markdown(f"**{x["author"]}: [{x["title"]}](https://orgelpredigt.ur.de/{x["id"]})**")
            for quote in quote_dict[x["id"]]:
                st.html(f'<div class="full_quote"><div class="quote-box">{quote}</div></div>')

    st.header("Verteilung der Nachnutzung über die Quelle hinweg")

    st.header("Akkumulierte Verteilung der Zitate über alle Predigten")
    quote_time_dist = st.selectbox(
            label="Zeitliche Einteilung",
            options=["ganzer Zeitraum", "50-Jahr-Intervalle", "25-Jahr-Intervalle"],
            placeholder="ganzer Zeitraum"
    )

    quote_type = "musikwerk" if selected_id.startswith("E10") else "orgelpredigt"
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

    if songtext:
        st.header(f"Verteilung der Zitate aus '{option}'")
        quote_distr = []
        for i in range(len(lines_cleaned)):
            quote_distr.append([i, lines_cleaned[i]["quotes"]])
        x_values = [item[0] for item in quote_distr]
        y_values = [item[1] for item in quote_distr]

        # Create bar plot
        fig = px.bar(x=x_values, y=y_values, labels={'x': 'Liedvers', 'y': 'zitierende Predigten'}, title=f"Zitate aus '{option}' je Vers")
        st.plotly_chart(fig)

    # umgebende zitate
    st.header("Umgebende Zitate")
    adjacent_quotes = get_shared_quotes(searchkey, ids)
    adjacent_quotes = [x.split("-")[0] for x in adjacent_quotes]
    adjacent_quotes = [get_short_info(x) if is_id(x) else x for x in adjacent_quotes]
    adjacent_quotes = [x.split(":")[1] if len(x.split(":")) > 1 else x for x in adjacent_quotes]

    counts = Counter(adjacent_quotes)
    top20 = counts.most_common(20)

    # Sort by frequency (descending)
    sorted_items = counts.most_common()
    labels = [item[0] for item in top20]
    values = [item[1] for item in top20]

    # Plot with Plotly
    fig = px.bar(
        x=labels, 
        y=values, 
        labels={'x': 'Quelle', 'y': 'Häufigkeit'},
        title=f"20 am häufigsten mit '{option}' im selben Paragraph erscheinenden Quellen/Lieder"
    )
    st.plotly_chart(fig)

    #####################
    ### NETWORK GRAPH ###
    #####################
    if len(sermon_info) > 1:
        def prune_by_total_degree(G, min_degree=2):
            while True:
                to_remove = [n for n, d in G.degree() if d < min_degree]
                if not to_remove:
                    break
                G.remove_nodes_from(to_remove)
            return G

        graph_node_kind = st.selectbox(
                    label="Zitattyp",
                    options=["Alle", "Musikwerke", "Quellen/Literatur", "Orgelpredigten"],
            )

        node_map = {"Alle": "E", "Musikwerke": "E10", 
                    "Quellen/Literatur": ("E08", "E09"), 
                    "Orgelpredigten": "E00"}

        network = []
        for id in quoting_sermons:
            item = {}
            current_sermon = Sermon(id)
            item["id"] = current_sermon.id
            item["links"] = [item for item in current_sermon.all_references if (is_id(item) and item.startswith(node_map[graph_node_kind]))]
            network.append(item)

        ##### Sermons and Sources
        # build raw (source, target) list
        connections = [(sermon['id'], link) for sermon in network for link in sermon['links']]

        # count incoming references per target
        target_counts = Counter(tgt for _, tgt in connections)

        # keep only edges where the TARGET is referenced >= 2 times
        filtered_connections = [(src, tgt) for src, tgt in connections if target_counts[tgt] >= 2]

        G2 = nx.DiGraph()
        G2.add_edges_from(filtered_connections) 
        G2 = prune_by_total_degree(G2, min_degree=2)

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
                            text="<br>Zitatnetzwerk zwischen Predigten, Musikwerken und Literatur",
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

        for src, tgt in G2.edges():
            x0, y0 = pos[src]
            x1, y1 = pos[tgt]
            sermons_sources_network.add_annotation(
                ax=x0, ay=y0,
                x=x1, y=y1,
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=3,  # arrow style
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="#888"
            )

        st.plotly_chart(sermons_sources_network)