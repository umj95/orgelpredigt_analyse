import sys
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print(">>> SYS.PATH:", sys.path)  # (optional, for debugging)


import streamlit as st
from core.utils import Sermon, get_short_info

from collections import Counter

import plotly.express as px

import matplotlib.pyplot as plt

import plotly.graph_objects as go
import pandas as pd

import folium
import json
import re
from pathlib import Path

# root directory path
ROOT = Path(__file__).resolve().parents[2]

print("WORKING DIR:", os.getcwd())
print("SYS.PATH:", sys.path)

st.set_page_config(
    page_title="Orgelpredigt_Analyse",
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

#########################
##### CHOOSE SERMON #####
#########################

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

### Streamlit
st.markdown("Welche Predigt soll analysiert werden? Bitte ID eingeben oder Predigt aus dropdown-Menü auswählen")
col1, col2 = st.columns([0.8, 0.2])

with col1:
    ids = []
    for i in relevant_sermons:
        ids.append(f"{i[1]} -- {i[0]}")

    option = st.selectbox(
        label="Predigt auswählen",
        options=ids,
        placeholder="Predigttitel -- Predigt-ID"
    )
with col2:
    input_id = st.text_input("oder  Predigt-ID eingeben")

if input_id:
    ids = [item[0] for item in relevant_sermons]
    if input_id in ids:
        sermon = Sermon(input_id)
    else:
        st.error("Die Eingabe kann keiner edierten Predigt zugewiesen werden.")
else:
    sermon = Sermon(option[-7:])

if Path(ROOT / f"predictions/{sermon.id}_predictions.json").is_file():
    # file exists:
    with open(ROOT / f"predictions/{sermon.id}_predictions.json", "r", encoding="utf-8") as f:
        predictions = json.load(f)
else:
    predictions = []


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
map = folium.Map(location=[50.8, 8.7], zoom_start=5)

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

text_types_piechart = px.pie(values=data, 
                             names=labels, 
                             title='Anteile der Zitate am Gesamttext', 
                             color=labels,
                             color_discrete_map=color_map)

##### list of quotations
def generate_normalized_gradient(rgb, n):
    """
    Generate a list of `n` normalized RGB gradient values
    that fade from black to the input `rgb` color.

    Parameters:
        rgb (tuple): A tuple of 3 integers (R, G, B), each 0-255.
        n (int): Number of gradient steps.

    Returns:
        list of tuples: Each tuple contains normalized (R, G, B) values.
    """
    rgb = rgb[4:-1]
    rgb = tuple(int(x) for x in rgb.split(", "))
    if not (isinstance(rgb, tuple) and len(rgb) == 3 and all(0 <= val <= 255 for val in rgb)):
        raise ValueError("RGB must be a tuple of three integers between 0 and 255.")
    if n <= 0:
        raise ValueError("Number of gradient steps must be positive.")
    
    gradient = []
    for i in range(n):
        ratio = i / (n - 1) if n > 1 else 1
        r = (rgb[0] * ratio) / 255
        g = (rgb[1] * ratio) / 255
        b = (rgb[2] * ratio) / 255
        gradient.append((r, g, b))
    
    return gradient

lit_labels = []
lit_data = []
lit_titel = []
lit_wordshare = []
lit_wordfraction = []
orgel_labels = []
orgel_data = []
orgel_titel = []
orgel_wordshare = []
orgel_wordfraction = []
musik_labels = []
musik_data = []
musik_titel = []
musik_wordshare = []
musik_wordfraction = []
for quelle in sermon.literaturzitate:
    lit_titel.append(str(quelle["item"]))
    lit_wordshare.append(quelle["word_share"])
    lit_wordfraction.append(float(f"{(quelle['word_share']/len(sermon.words)*100):.2f}"))
    lit_labels.append(str(quelle["item"]))
    lit_data.append(quelle["word_share"])
for predigt in sermon.orgelpredigtzitate:
    orgel_titel.append(str(predigt["item"]))
    orgel_wordshare.append(predigt["word_share"])
    orgel_wordfraction.append(float(f"{(predigt['word_share']/len(sermon.words)*100):.2f}"))
    orgel_labels.append(str(predigt["item"]))
    orgel_data.append(predigt["word_share"])
for musik in sermon.musikzitate:
    musik_titel.append(str(musik["item"]))
    musik_wordshare.append(musik["word_share"])
    musik_wordfraction.append(float(f"{(musik['word_share']/len(sermon.words)*100):.2f}"))
    musik_labels.append(str(musik["item"]))
    musik_data.append(musik["word_share"])


labels = []
values = []
colors = []
for item, broad_color in zip([[lit_labels, lit_data], 
                              [orgel_labels, orgel_data], 
                              [musik_labels, musik_data]], 
                             ['quelle', 'orgelpredigt', 'musikwerk']):
    print(item[0])
    #for x,y  in item[0], item[1]:
    labels += item[0]
    values += item[1]
    colors += generate_normalized_gradient(color_map[broad_color], len(labels))

quotations_piechart = go.Figure(go.Pie(values=values, 
                             labels=labels, 
                             marker=dict(colors=colors),
                             title='Verwendete Zitate'))

quotations_piechart.update_layout(
    width=700,
    height=700,
    margin=dict(t=80, b=50, l=50, r=50),
    title_x=0.5,  # Center title
    title='Verwendete Zitate',
    legend=dict(
        orientation="h",  # horizontal legend
        y=-0.1  # push legend below chart
    )
)

# create dataframe for table view
literatur = pd.DataFrame(
    {'Titel': lit_titel + orgel_titel + musik_titel,
     'Länge': lit_wordshare + orgel_wordshare + musik_wordshare,
     '% Anteil': lit_wordfraction + orgel_wordfraction + musik_wordfraction
    }).sort_values(by=['% Anteil'], ascending=False)
literatur['Titel'] = literatur['Titel'].apply(lambda x: ' '.join(x.split()[:20]))
literatur.style.hide()

##### quotation distribution over sermon in 100-Word-Chunks
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
        id_checker = re.match(r'E[01][0-9]{5}', str(ref))
        if id_checker:
            id = id_checker[0]
            print(id)
        else: 
            id = str(ref)
        name = f'{id}' if is_id(ref) else key_cleaned
        url=f'https://orgelpredigt.ur.de/{str(ref).strip()}' if is_id(ref) else ""
        quote_distribution_chunked.add_trace(go.Bar(
            name=name, 
            x=[bar_title], 
            y=[val],
            hovertemplate=f'<b>{get_short_info(id)}</b><br>Value: {val} Words<br>Link: <a href="{url}">{ref}</a><extra></extra>',
            marker_color=color_map.get(str(color).strip(), 'gray')
            ))

quote_distribution_chunked.update_layout(barmode='stack')

################################
##### TEXT WITH HIGHLIGHTS #####
################################

def sentence_to_html(sentence: dict, par_nr: int, sentence_nr: int, preds: list) -> str:
    """Takes a sentence dictionary and returns an html <p>-tag with appropriate child tags
        Args:
            sentence: The dict containing the keys "words" (list), "types" (list) and "references" (list of lists)
            par_nr: The number of the paragraph
            sentence_nr: The number of the sentence in the paragraph
        Returns:
            A string containing the tag
    """

    def add_tooltip(tooltips: list) -> str:
        tooltip = '<span class="tooltiptext">'
        print(tooltips)
        for elem in tooltips:
            if elem[0] == "machine":
                tooltip += f'<b>Maschinelle Auszeichnung</b><br/>'
            else:
                tooltip += f'<b>Manuelle Auszeichnung</b></br>'
            if isinstance(elem[1], str):
                if is_id(elem[1]):
                    tooltip += f'<a href="https://orgelpredigt.ur.de/{elem[1]}" target="_blank">{get_short_info(elem[1])}</a></br>'
                else:
                    tooltip += f'{elem[1]}</br>'
            else:
                tooltip += f'elem[1] is a list!</br>'
        tooltip += '</span>'
        return tooltip

    tag = f'<span class="orgelpredigt_span" id="{par_nr}-{sentence_nr}">'

    current_tag = []

    words = sentence["words"]
    types = sentence["types"]
    refs = sentence["references"]

    multiple_machine = ""
    machine_and_manual = ""
    tooltips = []

    # create spans for detected automatic tags
    pred_nr = len(preds)        # how many predictions exist for this sentence?
    if pred_nr > 0:
        if pred_nr > 1:
            multiple_machine = "multi_machine"
        for pred in preds:
            if pred["pred_type"] == "bibel":
                tooltip_add = "Lutherbibel "
            else:
                tooltip_add = "<b>Maschinelle Auszeichnung</b><br/>Crüger, <i>Praxis Pietatis Melica</i> S."
            tag += f'<span class="{pred["model"]} machine_{pred["pred_type"]} machine_tooltip {multiple_machine}">'
            
            tooltips.append(["machine", f"{tooltip_add}{pred["ref_id"]}: „{pred["text"]}“ (Ähnlichkeit: {pred["similarity"]})"])


    # create spans for manual tags
    if types[0] != "":
        if pred_nr > 0:
            machine_and_manual = "mach_and_man"
        current_tag.append(types[0].strip())
        tag += f'<span class="{types[0].strip()} {machine_and_manual} manual_tooltip">'
        for ref in set(flatten(refs)):
            tooltips.append(["manual", ref])
    
    # add tooltip if applicable
    if len(tooltips) > 0:
        tag += add_tooltip(tooltips)

    for word, type, ref in zip(words, types, refs):

        if len(ref) > 0:
            thisref = ref[-1]
        else:
            thisref = ''

        if type == "":
            if len(current_tag) == 0:
                tag += f' {word}'
            else:
                tag += f'</span> {word}'
                current_tag.pop()
        elif len(current_tag) > 0:
            if type.strip() == current_tag[-1]:
                tag += f' {word}'
            else:
                #tag += f'<span class="{type.strip()} manual_tooltip">{add_tooltip(thisref)}{word}'
                tag += f'<span class="{type.strip()} manual_tooltip">{word}'
                current_tag.append(type.strip())
        else:
            #tag += f'<span class="{type.strip()} manual_tooltip">{add_tooltip(thisref)}{word}'
            tag += f'<span class="{type.strip()} manual_tooltip">{word}'
            current_tag.append(type.strip())
    
    if len(current_tag) != 0:
        tag += '</span>'
    
    if pred_nr > 0:
        for i in range(pred_nr):
            tag += '</span>'

    return f'{tag}</span>'



##########################
##### STREAMLIT PAGE #####
##########################

def create_checkboxes(options):
    selected_options = {}
    for option in options:
        selected_options[option] = st.checkbox(option)
    return selected_options

st.title(f"{str(sermon)} – Analyse")

tab1, tab2 = st.tabs(["Überblick", "Predigttext"])
col1, col2 = st.columns(2, gap="small", vertical_alignment="top", border=False)

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

with tab1:
    with col1:
        ##### Geographischer Überblick
        st.header("Geographischer Überblick zu Predigt und Biographie des Autors")
        st.components.v1.html(folium.Figure().add_child(map).render(), height=500)
        #st.header("Zitate")
        st.plotly_chart(text_types_piechart)

    with col2:
        st.header("Verteilung von Zitaten im Text")
        st.plotly_chart(quote_distribution_chunked)

        ##### Überblick Zitate
        st.plotly_chart(quotations_piechart)

with tab2:
    human = False
    machine = False

    human = st.checkbox("Manuelle Annotationen")

    csv_list = []

    # create checkboxes for all machine annotations found
    if predictions:
        options = []
        files = {}
        types = {}
        for pred in predictions:
            identifier = f"Maschinelle Annotation – Typ: {pred["task"]}, Methode: {pred["method"]}, Unschärfe: {pred['fuzziness']}, Datum: {pred['date']}"

            files[identifier] = pred['file']
            types[identifier] = pred['task']
            options.append(identifier)

        selected = create_checkboxes(options)
        if any(selected.values()):
            machine = True
        for key, val in selected.items():
            if val == True:
                csv_list.append([pd.read_csv(files[key]), types[key]])

    st.markdown(f"""
            <style>
                div.orgelpredigt {{
                    padding: 10%;
                }}
                div.parmarker {{
                    margin-left: -8em;
                    color: lightgrey;
                }}
                span.mach_and_man {{
                }}
                a {{
                    color: skyblue;
                }}
            </style>
                """, unsafe_allow_html=True)

    if human:
        st.markdown(f"""
            <style>
                span.musikwerk {{
                    font-style: italic;
                    text-decoration: underline {color_map["musikwerk"]} 2px;
                    border-radius: 5px; 
                    padding: 2px; 
                }}
                span.orgelpredigt {{
                    font-style: italic;
                    text-decoration: underline {color_map["orgelpredigt"]} 2px;
                    border-radius: 5px; 
                    padding: 2px; 
                }}
                span.literatur {{
                    font-style: italic;
                    text-decoration: underline {color_map["literatur"]} 2px;
                    border-radius: 5px; 
                    padding: 2px; 
                }}
                span.quelle {{
                    font-style: italic;
                    text-decoration: underline {color_map["quelle"]} 2px;
                    border-radius: 5px; 
                    padding: 2px; 
                }}
                span.bibel {{
                    font-style: italic;
                    text-decoration: underline {color_map["bibel"]} 2px;
                    border-radius: 5px; 
                    padding: 2px; 
                }}
                /* Tooltip container */
                .manual_tooltip {{
                    position: relative;
                    display: inline-block;
                }}
                /* Tooltip text */
                .manual_tooltip .tooltiptext {{
                    font-weight: normal;
                    width: 120px;
                    background-color: #555;
                    color: #fff;
                    text-align: center;
                    padding: 5px 0;
                    border-radius: 6px;

                    /* Position the tooltip text */
                    position: absolute;
                    z-index: 1;
                    bottom: 125%;
                    left: 50%;
                    margin-left: -60px;

                    /* Fade in tooltip */
                    opacity: 0;
                    transition: opacity 0.3s;
                }}

                /* Tooltip arrow */
                .manual_tooltip .tooltiptext::after {{
                    content: "";
                    position: absolute;
                    top: 100%;
                    left: 50%;
                    margin-left: -5px;
                    border-width: 5px;
                    border-style: solid;
                    border-color: #555 transparent transparent transparent;
                }}

                /* Show the tooltip text when you mouse over the tooltip container */
                .manual_tooltip:hover .tooltiptext {{
                    visibility: visible;
                    opacity: 1;
                }} 
            </style>
            """, unsafe_allow_html=True)
    
    if machine:
        st.markdown(f"""
            <style>
                span.machine_lieder {{
                    background-color: {color_map["musikwerk"]}; 
                    border-radius: 5px; 
                    padding: 2px; 
                }}
                span.machine_orgelpredigt {{
                    background-color: {color_map["orgelpredigt"]}; 
                    border-radius: 5px; 
                    padding: 2px; 
                }}
                span.machine_literatur {{
                    background-color: {color_map["literatur"]}; 
                    border-radius: 5px; 
                    padding: 2px; 
                }}
                span.machine_quelle {{
                    background-color: {color_map["quelle"]}; 
                    border-radius: 5px; 
                    padding: 2px; 
                }}
                span.machine_bibel {{
                    background-color: {color_map["bibel"]}; 
                    /*border: 5px solid black;*/
                    border-radius: 5px; 
                    padding: 2px; 
                }}
                span.multi_machine {{
                    background: linear-gradient(90deg,rgba(131, 58, 180, 1) 0%, rgba(253, 29, 29, 1) 50%, rgba(252, 176, 69, 1) 100%);
                }}
                /* Tooltip container */
                .machine_tooltip {{
                    position: relative;
                    display: inline-block;
                }}
                /* Tooltip text */
                .machine_tooltip .tooltiptext {{
                    width: 120px;
                    background-color: #555;
                    color: #fff;
                    text-align: center;
                    padding: 5px 0;
                    border-radius: 6px;

                    /* Position the tooltip text */
                    position: absolute;
                    z-index: 1;
                    bottom: 125%;
                    left: 50%;
                    margin-left: -60px;

                    /* Fade in tooltip */
                    opacity: 0;
                    transition: opacity 0.3s;
                }}

                /* Tooltip arrow */
                .machine_tooltip .tooltiptext::after {{
                    content: "";
                    position: absolute;
                    top: 100%;
                    left: 50%;
                    margin-left: -5px;
                    border-width: 5px;
                    border-style: solid;
                    border-color: #555 transparent transparent transparent;
                }}

                /* Show the tooltip text when you mouse over the tooltip container */
                .machine_tooltip:hover .tooltiptext {{
                    visibility: visible;
                    opacity: 1;
                }} 
            </style>
            """, unsafe_allow_html=True)
    if not human:
        st.markdown(f"""
            <style>
                .manual_tooltip .tooltiptext{{
                    display: none;
                }}
            </style>
        """, unsafe_allow_html=True)
    if not machine:
        st.markdown(f"""
            <style>
                .machine_tooltip .tooltiptext{{
                    display: none;
                }}
            </style>
        """, unsafe_allow_html=True)

    sermon_html = f'<div class="orgelpredigt">'

    for i in range(len(sermon.chunked)):
        #create a new div for each paragraph
        paragraph_text = f'<div class="parmarker">Paragraph {i}</div><p class="orgelpredigt_p" id="{i}">'
        # go over each sentence
        for j in range(len(sermon.chunked[i])):
            # see if any predictions apply and put them in a list of dicts
            preds = []
            task = "lieder"
            for df, task in csv_list:
                row = df[(df['Paragraph'] == i) & (df['Satz'] == j)]
                if not row.empty:
                    row_dict = row.iloc[0].to_dict()
                    if "Bibelstelle" in row:
                        row_dict["model"] = "similarity_search"
                        row_dict["pred_type"] = task
                        row_dict["ref_id"] = row_dict["Fundstelle"]
                        row_dict["text"] = row_dict["Vers"]
                        row_dict["similarity"] = row_dict["Ähnlichkeit"]
                    else:
                        row_dict["model"] = "similarity_search"
                        row_dict["pred_type"] = task
                        row_dict["ref_id"] = row_dict["Fundstelle"]
                        row_dict["text"] = row_dict["Vers"]
                        row_dict["similarity"] = row_dict["Ähnlichkeit"]
                    preds.append(row_dict)

            paragraph_text += sentence_to_html(sermon.chunked[i][j], i, j, preds)
        paragraph_text += "</p>"
        sermon_html += paragraph_text

    sermon_html += "</div>"

    st.header(sermon.volltitel)
    st.html(sermon_html)


