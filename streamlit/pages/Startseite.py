import sys
import os

# Get the absolute path to the repository root
from pathlib import Path

# root directory path
root = Path(os.getcwd()).resolve().parents[1]

# Add the repository root to the Python path
sys.path.append(str(root))

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from core.utils import Sermon, Person, get_short_info
from collections import Counter

import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx

import plotly.graph_objects as go
import pandas as pd

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

st.title('Zitatnetzwerke in deutschsprachigen Orgelpredigten der frühen Neuzeit')

col1, col2 = st.columns([0.8,0.2], gap="small", vertical_alignment="top", border=False)

with col1:
    st.markdown("""
Diese Seite bietet einen interaktiven Einstieg in die textuellen Netzwerke in welchen sich frühmoderne Orgelpredigten bewegen. Die Daten, auf denen diese Ansicht beruht, basieren auf dem DFG-geförderten Projekt [Deutsche Orgelpredigtdrucke zwischen 1600 und 1800 – Katalogisierung, Texterfassung, Auswertung](https://orgelpredigt.ur.de/). Als dynamische Oberfläche zur Analyse der Netzwerkeffekte in entweder einzelnen oder mehreren Orgelpredigten bildet diese Seite Teil der Masterarbeit _‚Das Wort sie sollen lassen stahn‘: Digitale Analyse intertextueller Beziehungen in historischen Textcorpora anhand von Liedzitaten in deutschen Orgelpredigten_. Die hier versammelten Visualisierungen befassen sich - neben der Vermittlung grundlegender statistischer Eigenschaften des Korpus - vor allem mit drei Themen:

1. Wie verteilen sich Zitate aus anderen Predigten, Literatur und Musikwerken innerhalb einer einzelnen Predigt, sowie über eine Vielzahl an Predigten?
2. Welche Passagen aus besonders vielzitierten Liedern werden besonders gerne zitiert, an welchen Stellen in den Predigten, und in welchem Zeitraum?
3. Welche Zitate im Predigttext werden maschinell in einem Vergleichskorpus erkannt, wie unterscheiden sich hier Identifikationen basierend auf string-Vergleich und basierend auf einer Vektorrepräsentation des Textes? Wie überschneiden sie sich untereinander und mit den manuell markierten Daten?""")
    st.subheader("Manuell und maschinell erhobene Daten")
    st.markdown("""
Die hier aufbereiteten Daten greifen primär auf die von Forschenden (allen voran Dr. Lucinde Braun) im Rahmen des Projektes vorgenommenen manuellen Auszeichnungen der Predigttexte zurück. Einzig in der Anzeige der maschinell generierten Zitate werden (natürlich) automatisch erkannte Zitate angezeigt. Für dieses Experiment wurde in den individuellen Predigttexten nur nach Zitaten aus der Lutherbibel (Aufbereitet aus folgenden Quellen: https://sourceforge.net/projects/zefania-sharp/files/Bibles/GER/Lutherbibel/Luther%201545%20%28Letzte%20Hand%29/ (altes Testament) und https://www.deutschestextarchiv.de/book/show/luther_septembertestament_1522 (neues Testament)) und evangelischen Kirchenliedern in Johann Crügers _Praxis Pietatis Melica_ (aufbereitet aus folgender Quelle: https://www.digitale-sammlungen.de/en/view/bsb10589853) gesucht. Der phrasenweise Vergleich von Predigttext und Bibel bzw. Gesangbuch wurde sowohl anhand von strings, also Buchstabensequenzen, und Vektoren, also der Repräsentation der Texte als Zahlenreihen, durchgeführt, die Auswahl, welche der auf die eine oder andere Art klassifizierten Zitate angezeigt werden sollen, kann jeweils individuell vorgenommen werden.

Die maschinelle Suche nach Zitaten erfolgte automatisch, mit voreingestellten Toleranzwerten für Abweichungen der strings (0,17) bzw. Vektoren (0,60), die in Kapitel 6 der Masterarbeit eingehender besprochen werden. Als Fallbeispiel für die Auswirkung abweichender Toleranzwerte dient Raphael Skubowius' Predigt (ID: E000036), bei der mehr Daten die anhand anderer Werte erhoben wurden zur Verfügung stehen.""")
    st.html("""
<p>Die Darstellung der Zitate erfolgt nach folgendem Muster: <span style="font-style: italic; text-decoration: underline rgb(192, 54, 157) 2px;">Manuell ausgezeichnete Zitate erscheinen unterstrichen und kursiv.</span> <span style="background-color: rgb(246, 169, 122, 0.8974); color: inherit; border-radius: 5px; padding: 2px;">Maschinell erkannte Zitate erscheinen hinterlegt,</span> <span style="background-color: rgb(246, 169, 122, 0.6); color: inherit; border-radius: 5px; padding: 2px;">wobei die Farbintensität die Sicherheit der Zuschreibung</span> <span style="background-color: rgb(246, 169, 122, 0.4); color: inherit; border-radius: 5px; padding: 2px;">durch den Algorithmus repräsentiert.</span> <span style="background-image: linear-gradient(to right,rgba(246, 169, 122, 0.8696) 0%, rgba(192, 54, 157, 0.8511) 100%); border-radius: 5px; padding: 2px;">Zitate, die maschinell mehereren möglichen Quellen zugeschrieben werden können, erscheinen mit gradient hinterlegt.</span></p>
<p>Über alle Visualisierungen hinweg sind Farben mit Zitattypen assoziiert: <span style="background-color: rgb(246, 169, 122); color: inherit; border-radius: 5px; padding: 2px;">Bibelzitate sind orange</span>, Zitate aus <span style="background-color: rgb(250, 120, 118); color: inherit; border-radius: 5px; padding: 2px;">Quellen</span> und <span style="background-color: rgb(234, 79, 136); color: inherit; border-radius: 5px; padding: 2px;">Literatur</span> sind hell- bzw. dunkelrot</span>, <span style="background-color: rgb(192, 54, 157); color: white; border-radius: 5px; padding: 2px;">Zitate aus Musikwerken sind lila</span> und <span style="background-color: rgb(135, 44, 162); color: white; border-radius: 5px; padding: 2px;">Zitate aus Orgelpredigten sind violett</span>. Der Farbverlauf soll hier auch die relative Häufigkeit der jeweiligen Zitattypen wiedergeben. Die verhältnismäßig große Ähnlichkeit zwischen Quellen und Literatur ist bewusst gewählt und spiegelt die große Ähnlichkeit dieser beiden Tags in ihrer Verwendung im Predigtmarkup wieder.</p>
""")
    
    st.markdown("""
Grundsätzlich ist zu beachten, dass, während die zu den Predigten gelieferten Metadaten auf den entsprechenden Datensätzen des Projektes basieren, die Angaben zu Zitationen in Predigten _allein auf im Fließtext markierten Passagen_ beruhen, also nicht auf Marginalien oder Erwähnungen in Fußnoten oder Einführungstexten. Aufgrund dessen können sich die hier präsentierten Daten teils anders ausnehmen als in der Darstellung im Orgelpredigt-Portal.""")
    
    st.subheader("Zur Nutzung")
    st.markdown("""Die Startseite vermittelt einen generellen Überblick. Auf der Seite 'Orgelpredigt Analyse' können einzelne Predigten genauer inspiziert und im Volltext angezeigt werden. Über 'Orgelpredigt Vergleich' können zwei oder mehr Predigten gemeinsam auf Überschneidungen hin untersucht werden. 

Diese Seite ist als Prototyp konzipiert und in keiner Weise optimiert. Daher können Berechnungen, v.a. für kumulative Plots, unter Umständen etwas Zeit in Anspruch nehmen. Die Animation oben rechts im Fenster zeigt an, wenn alle Operationen abgeschlossen sind.
""")

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
###### STATISTICS #######
#########################

sermons_per_year = [[1602,
  'Christliche Predigt (Tübingen 1602)',
  '<a href="https://orgelpredigt.ur.de/E000001" target="_blank">Christliche Predigt (Tübingen 1602)</a>'],
 [1605,
  'Musica instrumentalis (Meißen 1605)',
  '<a href="https://orgelpredigt.ur.de/E000002" target="_blank">Musica instrumentalis (Meißen 1605)</a>'],
 [1606,
  'Christliche Predigt (Tübingen 1606)',
  '<a href="https://orgelpredigt.ur.de/E000029" target="_blank">Christliche Predigt (Tübingen 1606)</a>'],
 [1610,
  'Elogium Organi Musici (Altenburg 1610)',
  '<a href="https://orgelpredigt.ur.de/E000030" target="_blank">Elogium Organi Musici (Altenburg 1610)</a>'],
 [1621,
  'Corona Templi (Nürnberg 1621)',
  '<a href="https://orgelpredigt.ur.de/E000099" target="_blank">Corona Templi (Nürnberg 1621)</a>'],
 [1624,
  'Vlmische Orgel Predigt (Ulm 1624)',
  '<a href="https://orgelpredigt.ur.de/E000003" target="_blank">Vlmische Orgel Predigt (Ulm 1624)</a>'],
 [1628,
  'Musica ecclesiastica (Stettin 1628)',
  '<a href="https://orgelpredigt.ur.de/E000098" target="_blank">Musica ecclesiastica (Stettin 1628)</a>'],
 [1647,
  'Kostbare Bosische Orgel (Zwickau 1647)',
  '<a href="https://orgelpredigt.ur.de/E000096" target="_blank">Kostbare Bosische Orgel (Zwickau 1647)</a>'],
 [1648,
  'Längst=gewüntzschte Mittweidische Orgel=Freude (Dresden 1648)',
  '<a href="https://orgelpredigt.ur.de/E000095" target="_blank">Längst=gewüntzschte Mittweidische Orgel=Freude (Dresden 1648)</a>'],
 [1651,
  'Organologismos (Dresden 1651)',
  '<a href="https://orgelpredigt.ur.de/E000092" target="_blank">Organologismos (Dresden 1651)</a>'],
 [1652,
  'Stolpenische Ehren-Crone (Dresden 1652)',
  '<a href="https://orgelpredigt.ur.de/E000091" target="_blank">Stolpenische Ehren-Crone (Dresden 1652)</a>'],
 [1660,
  'Organolustria Evangelico-Stambachiana (Hof 1660)',
  '<a href="https://orgelpredigt.ur.de/E000090" target="_blank">Organolustria Evangelico-Stambachiana (Hof 1660)</a>'],
 [1664,
  'Encoenia HierOrganica (Halle 1664)',
  '<a href="https://orgelpredigt.ur.de/E000089" target="_blank">Encoenia HierOrganica (Halle 1664)</a>'],
 [1666,
  'Orgel=Predigt (Arnstadt 1666)',
  '<a href="https://orgelpredigt.ur.de/E000086" target="_blank">Orgel=Predigt (Arnstadt 1666)</a>'],
 [1667,
  'Das fröliche Hallelujah (Halle 1667)',
  '<a href="https://orgelpredigt.ur.de/E000085" target="_blank">Das fröliche Hallelujah (Halle 1667)</a>'],
 [1671,
  'Das Gott=Lob=Schallende Hosianna (Leipzig 1671)',
  '<a href="https://orgelpredigt.ur.de/E000083" target="_blank">Das Gott=Lob=Schallende Hosianna (Leipzig 1671)</a>'],
 [1672,
  'Geistliches Orgelwerk (Erfurt 1672)',
  '<a href="https://orgelpredigt.ur.de/E000082" target="_blank">Geistliches Orgelwerk (Erfurt 1672)</a>'],
 [1673,
  'Denck- und Danck-Säule (Rothenburg ob der Tauber [1673])',
  '<a href="https://orgelpredigt.ur.de/E000079" target="_blank">Denck- und Danck-Säule (Rothenburg ob der Tauber [1673])</a>'],
 [1675,
  'Das fröliche Halleluja (Wittenberg 1675)',
  '<a href="https://orgelpredigt.ur.de/E000078" target="_blank">Das fröliche Halleluja (Wittenberg 1675)</a>'],
 [1676,
  'Das Lieblich=klingende Orgeln und Saiten=Spiel (Coburg 1676)',
  '<a href="https://orgelpredigt.ur.de/E000075" target="_blank">Das Lieblich=klingende Orgeln und Saiten=Spiel (Coburg 1676)</a>'],
 [1676,
  'Die andere Predigt (Coburg 1676)',
  '<a href="https://orgelpredigt.ur.de/E000106" target="_blank">Die andere Predigt (Coburg 1676)</a>'],
 [1680,
  'Geistlich= und Gott wohlgefälliges Lob- und Danck-Opffer (Bayreuth 1680)',
  '<a href="https://orgelpredigt.ur.de/E000073" target="_blank">Geistlich= und Gott wohlgefälliges Lob- und Danck-Opffer (Bayreuth 1680)</a>'],
 [1681,
  'Gott und Gnug (Meißen 1681)',
  '<a href="https://orgelpredigt.ur.de/E000072" target="_blank">Gott und Gnug (Meißen 1681)</a>'],
 [1683,
  'Cithara Theologica (Schleusingen 1683)',
  '<a href="https://orgelpredigt.ur.de/E000070" target="_blank">Cithara Theologica (Schleusingen 1683)</a>'],
 [1685,
  'Organi Laudes (Plauen 1685)',
  '<a href="https://orgelpredigt.ur.de/E000108" target="_blank">Organi Laudes (Plauen 1685)</a>'],
 [1686,
  'Organum Mysticum (Dresden 1686)',
  '<a href="https://orgelpredigt.ur.de/E000069" target="_blank">Organum Mysticum (Dresden 1686)</a>'],
 [1687,
  'Das dem Allmächtigen abzustattende Lob (Altenburg s.a.)',
  '<a href="https://orgelpredigt.ur.de/E000068" target="_blank">Das dem Allmächtigen abzustattende Lob (Altenburg s.a.)</a>'],
 [1689,
  'Organo-Praxis Mystica (Görlitz 1689)',
  '<a href="https://orgelpredigt.ur.de/E000067" target="_blank">Organo-Praxis Mystica (Görlitz 1689)</a>'],
 [1695,
  'Eine Christliche Orgel=Predigt (Danzig 1695)',
  '<a href="https://orgelpredigt.ur.de/E000065" target="_blank">Eine Christliche Orgel=Predigt (Danzig 1695)</a>'],
 [1696,
  'Schuldiges Lob Gottes (Nürnberg 1696)',
  '<a href="https://orgelpredigt.ur.de/E000063" target="_blank">Schuldiges Lob Gottes (Nürnberg 1696)</a>'],
 [1700,
  'Die Christliche Harmonie (Jena 1700)',
  '<a href="https://orgelpredigt.ur.de/E000060" target="_blank">Die Christliche Harmonie (Jena 1700)</a>'],
 [1704,
  'Christliche Orgel-Predigt (Danzig s.a.)',
  '<a href="https://orgelpredigt.ur.de/E000058" target="_blank">Christliche Orgel-Predigt (Danzig s.a.)</a>'],
 [1704,
  'Einweihungs-Predigt (Görlitz 1704)',
  '<a href="https://orgelpredigt.ur.de/E000059" target="_blank">Einweihungs-Predigt (Görlitz 1704)</a>'],
 [1709,
  'Orgel Weih-Predigt (Ansbach 1709)',
  '<a href="https://orgelpredigt.ur.de/E000056" target="_blank">Orgel Weih-Predigt (Ansbach 1709)</a>'],
 [1709,
  'Das rein-gestimmte Orgel-Werk unsers Herzens (Nürnberg s.a.)',
  '<a href="https://orgelpredigt.ur.de/E000057" target="_blank">Das rein-gestimmte Orgel-Werk unsers Herzens (Nürnberg s.a.)</a>'],
 [1711,
  'Davids Vermahnung (Dresden 1711)',
  '<a href="https://orgelpredigt.ur.de/E000055" target="_blank">Davids Vermahnung (Dresden 1711)</a>'],
 [1711,
  'Evangelischer Christen Gott-gefällige Kirch-Weyhung (Dresden 1711)',
  '<a href="https://orgelpredigt.ur.de/E000104" target="_blank">Evangelischer Christen Gott-gefällige Kirch-Weyhung (Dresden 1711)</a>'],
 [1720,
  'Vivum Dei Organum (Schneeberg s.a.)',
  '<a href="https://orgelpredigt.ur.de/E000053" target="_blank">Vivum Dei Organum (Schneeberg s.a.)</a>'],
 [1721,
  'Die Kneiphöffsche laute Orgel=Stimme (Königsberg 1721)',
  '<a href="https://orgelpredigt.ur.de/E000051" target="_blank">Die Kneiphöffsche laute Orgel=Stimme (Königsberg 1721)</a>'],
 [1721,
  'Ein wolgerührtes Orgel=Werck (Königsberg 1721)',
  '<a href="https://orgelpredigt.ur.de/E000052" target="_blank">Ein wolgerührtes Orgel=Werck (Königsberg 1721)</a>'],
 [1721,
  'Glaubiger Kinder Gottes Gott=gefällige Music (Augsburg 1721)',
  '<a href="https://orgelpredigt.ur.de/E000074" target="_blank">Glaubiger Kinder Gottes Gott=gefällige Music (Augsburg 1721)</a>'],
 [1726,
  'Die verstimmte Zwölff Grösseste Pfeiffen (Tübingen s.a.)',
  '<a href="https://orgelpredigt.ur.de/E000048" target="_blank">Die verstimmte Zwölff Grösseste Pfeiffen (Tübingen s.a.)</a>'],
 [1727,
  'Die edle und wohlgeordnete Music der Gläubigen (Halle 1727)',
  '<a href="https://orgelpredigt.ur.de/E000046" target="_blank">Die edle und wohlgeordnete Music der Gläubigen (Halle 1727)</a>'],
 [1728,
  'Hymnosophia sacra (Billwerder 1728)',
  '<a href="https://orgelpredigt.ur.de/E000045" target="_blank">Hymnosophia sacra (Billwerder 1728)</a>'],
 [1730,
  'Einweihungs-Predigt (Berlin 1730)',
  '<a href="https://orgelpredigt.ur.de/E000061" target="_blank">Einweihungs-Predigt (Berlin 1730)</a>'],
 [1735,
  'Das Neue Lied (Freiberg 1735)',
  '<a href="https://orgelpredigt.ur.de/E000042" target="_blank">Das Neue Lied (Freiberg 1735)</a>'],
 [1737,
  'Stimme des Predigers (1737)',
  '<a href="https://orgelpredigt.ur.de/E000109" target="_blank">Stimme des Predigers (1737)</a>'],
 [1739,
  'Die Billige Orgel-Freude (Danzig 1739)',
  '<a href="https://orgelpredigt.ur.de/E000041" target="_blank">Die Billige Orgel-Freude (Danzig 1739)</a>'],
 [1740,
  'Winnedisches Reminiscere (Stuttgart 1740)',
  '<a href="https://orgelpredigt.ur.de/E000039" target="_blank">Winnedisches Reminiscere (Stuttgart 1740)</a>'],
 [1747,
  'Den rechtmäßigen Gebrauch der Music (Königsberg 1747)',
  '<a href="https://orgelpredigt.ur.de/E000038" target="_blank">Den rechtmäßigen Gebrauch der Music (Königsberg 1747)</a>'],
 [1749,
  'Die heilige Sabbaths-Lust an dem Herrn (Danzig 1749)',
  '<a href="https://orgelpredigt.ur.de/E000036" target="_blank">Die heilige Sabbaths-Lust an dem Herrn (Danzig 1749)</a>'],
 [1749,
  'Christliche Predigt (Straßburg 1749)',
  '<a href="https://orgelpredigt.ur.de/E000037" target="_blank">Christliche Predigt (Straßburg 1749)</a>'],
 [1751,
  'Musicalische Orgel= Lob= und Ehren=Predigt (s.l. 1751)',
  '<a href="https://orgelpredigt.ur.de/E000035" target="_blank">Musicalische Orgel= Lob= und Ehren=Predigt (s.l. 1751)</a>'],
 [1753,
  'Lob= und Danck=Predigt (Berlin 1753)',
  '<a href="https://orgelpredigt.ur.de/E000034" target="_blank">Lob= und Danck=Predigt (Berlin 1753)</a>'],
 [1761,
  'Der Christen gerechte Freude (Breslau 1761)',
  '<a href="https://orgelpredigt.ur.de/E000027" target="_blank">Der Christen gerechte Freude (Breslau 1761)</a>'],
 [1765,
  'Die heiligen Verrichtungen in dem Hause des Herrn (Eisenach 1765)',
  '<a href="https://orgelpredigt.ur.de/E000024" target="_blank">Die heiligen Verrichtungen in dem Hause des Herrn (Eisenach 1765)</a>'],
 [1766,
  'Der rechte Gebrauch der Orgeln (Altenburg 1766)',
  '<a href="https://orgelpredigt.ur.de/E000023" target="_blank">Der rechte Gebrauch der Orgeln (Altenburg 1766)</a>'],
 [1767,
  'Das heilige und fröliche Aufsehen (Tübingen 1767)',
  '<a href="https://orgelpredigt.ur.de/E000021" target="_blank">Das heilige und fröliche Aufsehen (Tübingen 1767)</a>'],
 [1770,
  'Predigt am Feste der Heimsuchung Mariae (Rostock 1770)',
  '<a href="https://orgelpredigt.ur.de/E000020" target="_blank">Predigt am Feste der Heimsuchung Mariae (Rostock 1770)</a>'],
 [1778,
  'Der Dienst der Orgeln (Jena 1778)',
  '<a href="https://orgelpredigt.ur.de/E000016" target="_blank">Der Dienst der Orgeln (Jena 1778)</a>'],
 [1781,
  'Gast-Predigt (Ulm 1781)',
  '<a href="https://orgelpredigt.ur.de/E000014" target="_blank">Gast-Predigt (Ulm 1781)</a>'],
 [1781,
  'Rede und Predigt bey Einweihung der neuen Orgel (Stockholm 1781)',
  '<a href="https://orgelpredigt.ur.de/E000015" target="_blank">Rede und Predigt bey Einweihung der neuen Orgel (Stockholm 1781)</a>'],
 [1795,
  'Predigt bey Einweyhung der Orgel (s.l. 1795)',
  '<a href="https://orgelpredigt.ur.de/E000009" target="_blank">Predigt bey Einweyhung der Orgel (s.l. 1795)</a>'],
 [1797,
  'Predigt Bey der Einweihung einer Orgel (Leipzig 1797)',
  '<a href="https://orgelpredigt.ur.de/E000008" target="_blank">Predigt Bey der Einweihung einer Orgel (Leipzig 1797)</a>'],
 [1798,
  'Predigt bey der feyerlichen Einweihung der neuen Orgel (Magdeburg 1798)',
  '<a href="https://orgelpredigt.ur.de/E000007" target="_blank">Predigt bey der feyerlichen Einweihung der neuen Orgel (Magdeburg 1798)</a>']]
quotes_per_year = [[1602, [0, 7, 0]],
 [1605, [0, 1, 0]],
 [1606, [0, 8, 5]],
 [1610, [0, 3, 1]],
 [1621, [0, 13, 1]],
 [1624, [0, 23, 0]],
 [1628, [0, 6, 1]],
 [1647, [1, 20, 8]],
 [1648, [2, 11, 8]],
 [1651, [1, 24, 7]],
 [1652, [2, 15, 5]],
 [1660, [1, 10, 1]],
 [1664, [0, 7, 4]],
 [1666, [0, 11, 3]],
 [1667, [0, 10, 8]],
 [1671, [0, 15, 4]],
 [1672, [0, 5, 12]],
 [1673, [3, 10, 4]],
 [1675, [3, 6, 5]],
 [1676, [2, 12, 13]],
 [1676, [0, 14, 8]],
 [1680, [3, 10, 7]],
 [1681, [0, 15, 18]],
 [1683, [1, 10, 16]],
 [1685, [2, 12, 1]],
 [1686, [0, 32, 7]],
 [1687, [1, 45, 1]],
 [1689, [0, 24, 1]],
 [1695, [1, 23, 3]],
 [1696, [1, 10, 0]],
 [1700, [0, 13, 10]],
 [1704, [0, 28, 3]],
 [1704, [2, 49, 3]],
 [1709, [1, 7, 1]],
 [1709, [0, 15, 5]],
 [1711, [2, 49, 15]],
 [1711, [0, 15, 2]],
 [1720, [2, 104, 5]],
 [1721, [0, 13, 0]],
 [1721, [3, 12, 11]],
 [1721, [2, 1, 1]],
 [1726, [0, 4, 5]],
 [1727, [0, 8, 6]],
 [1728, [0, 157, 8]],
 [1730, [0, 5, 6]],
 [1735, [0, 11, 17]],
 [1737, [0, 1, 6]],
 [1739, [2, 20, 3]],
 [1740, [0, 1, 3]],
 [1747, [0, 5, 9]],
 [1749, [1, 34, 25]],
 [1749, [0, 1, 2]],
 [1751, [0, 0, 0]],
 [1753, [0, 1, 1]],
 [1761, [1, 34, 7]],
 [1765, [0, 7, 3]],
 [1766, [0, 0, 1]],
 [1767, [0, 0, 3]],
 [1770, [0, 0, 2]],
 [1778, [0, 0, 0]],
 [1781, [0, 0, 0]],
 [1781, [0, 0, 6]],
 [1795, [0, 0, 0]],
 [1797, [0, 0, 1]],
 [1798, [0, 0, 3]]]

df = pd.DataFrame(sermons_per_year, columns=["year", "title", "id"])
# Create full year range
all_years = pd.DataFrame({"year": range(1600,1801)})

# Merge to ensure every year appears, fill missing with 0
df_full = pd.merge(all_years, df, on="year", how="left").fillna(0)

agg = df.groupby("year").agg({
    "title": lambda x: "<br>".join(x.astype(str)),
    "id": lambda x: ", ".join(x.astype(str)),
    "year": "count"
}).rename(columns={"year": "count"}).reset_index()

sermons_per_year = px.bar(agg, x='year', y='count',
             hover_data='id',
             color_discrete_sequence=['rgb(135, 44, 162)'])
sermons_per_year.update_layout(
    title="Pro Jahr veröffentlichte Orgelpredigten",
    xaxis_title="Jahr",
    yaxis_title="Anzahl"
)


df_quotes = pd.DataFrame(quotes_per_year, columns=["Jahr", "Anzahl"])
df_quotes[["orgelpredigt", "literatur", "musikwerk"]] = pd.DataFrame(df_quotes["Anzahl"].tolist(), index=df_quotes.index)
df_quotes = df_quotes.drop(columns="Anzahl")

# Melt to long format for plotly express
df_long = df_quotes.melt(id_vars="Jahr", value_vars=["orgelpredigt", "literatur", "musikwerk"],var_name="type", value_name="Anzahl")

# Create figure
quotes_per_year = go.Figure()

# Add line traces with custom colors
for t in df_long["type"].unique():
    subset = df_long[df_long["type"] == t]
    quotes_per_year.add_trace(
        go.Scatter(
            x=subset["Jahr"],
            y=subset["Anzahl"],
            mode="lines+markers",
            name=t,
            line=dict(color=color_map[t], width=2),
            marker=dict(size=8)
        )
    )

# Layout
quotes_per_year.update_layout(
    title="Anzahl der individuell zitierten Werke pro Jahr nach Typen unterteilt",
    xaxis_title="Jahr",
    yaxis_title="Anzahl"
)

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

st.title("Zeitlicher Überblick")
plotcol1, plotcol2 = st.columns([0.5, 0.5])
with plotcol1:
    st.plotly_chart(sermons_per_year)
with plotcol2:
    st.plotly_chart(quotes_per_year)

st.title("Literatur- und Liedzitate zwischen allen Orgelpredigten")
st.markdown("Der folgende Netzwerk-Graph visualisiert die Verweise in Predigttexten auf Literatur, Musikwerke, sowie andere Predigten. Der Mouseover-Effekt zeigt die jeweiligen Titel an. Auf der Seite [Orgelpredigt Vergleich](https://orgelpredigt-analyse.streamlit.app/Orgelpredigt_Vergleich) kann dieser Plot auch für eine oder meherere individuelle Predigten generiert werden.")
st.plotly_chart(sermons_sources_network)

st.title("Liedzitate – kumulativ und diachron betrachtet")
st.markdown("Dieser 'Barcode' repräsentiert die kumulative Verwendung von Zitaten (Typ auswählbar) je Predigtprozent. Je dunkler der Balken, desto mehr Zitate des jeweiligen Typs erscheinen in dem entsprechenden Prozent. Über das Auswahlfeld 'Zeitliche Einteilung' können auch mehrere zeitlich gestaffelte Diagramme generiert werden. Auf der Seite [Orgelpredigt Vergleich](https://orgelpredigt-analyse.streamlit.app/Orgelpredigt_Vergleich) kann dieser Plot auch für eine oder mehrere individuelle Predigten generiert werden. Auf der Seite [Quellen Analyse](https://orgelpredigt-analyse.streamlit.app/Quellen_Analyse) zeigt er für ausgewählte Lieder an, an welchen Stellen im Predigtkorpus sie zitiert werden.")

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
