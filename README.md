# Prototypische Analyse Intertextueller Netzwerke anhand des Korpus deutscher Orgelpredigtdrucke
Dieses Repositorium enthält Code der im Rahmen meiner Masterarbeit geschrieben wurde. Der Code ist explorativ und dient lediglich den für die Arbeit durchgeführten Tests und ihren Visualisierungen 

## Projektabstract
Die Untersuchung von Text-Text-Beziehungen, eine klassische Domäne der Geisteswissenschaften, ist seit Beginn der Disziplin auch fester Bestandteil der _Digital Humanities_. Diese Arbeit behandelt anhand der Zitate protestantischer Kirchenlieder in frühneuzeitlichen deutschen Orgelpredigtdrucken die Frage, wie ein an Julia Kristevas _Intertextualitätsbegriff_ angelehnter Analyse- und Visualisierungsansatz helfen kann, Anknüpfungspunkte für die Analyse intertextueller Systeme in homogenen, historischen, nur bedingt kanonisierten Textkorpora zu schaffen.

---

The study of text-to-text relationships, a classic domain of the humanities, has
from the inception of the discipline been a staple of the Digital Humanities as
well. This thesis uses quotes of protestant hymns in early modern German prints
of organ sermons to investigate, how Julia Kristeva’s concept of intertextuality
can help us develop tools and methods of analysis of intertextual systems in
homogenous, historical, and only partially canonised text corpora.

## Pythonmodule
Das Projekt wurde mit Python 3.12.11 entwickelt. Die entscheidenden Pythonmodule für Datenaufbereitung, Analyse und Visualisierung waren:
- beautifulsoup4 4.13.4
- chromadb 1.0.16
- folium 0.20.0
- langchain_core 0.3.76
- langchain_community 0.3.27
- pandas 2.3.0
- plotly 6.1.2
- rapidfuzz 3.13.0
- streamlit 1.46.0
Alle weiteren für die Reproduktion der Ergebnisse notwendigen Module können der Datei pyproject.toml entnommen werden. 

## Datensätze
Neben den Analyseskripten und der Webaufbereitung enthält dieses Repositorium auch die Zwecke dieser Arbeit aufbereiteten Datensätze der Quellen, die als Grundlage der Analyse dienen.
Diese Umfassen:
- Die Datensätze des DFG-Projekts [_Deutsche Orgelpredigtdrucke zwischen 1600 und 1800 – Katalogisierung, Texterfassung, Auswertung_](https://orgelpredigt.ur.de) 
- Das Alte Testament in der Übersetzung von Martin Luther, basierend auf der digitalen Edition des [Zefania XML Projektes](https://sourceforge.net/projects/zefania-sharp/files/Bibles/GER/Lutherbibel/Luther%201545%20%28Letzte%20Hand%29/)
- Das Neue Testament in der Übersetzung von Martin Luther, basierend auf der digitalen Edition des [Deutschen Textarchivs](https://www.deutschestextarchiv.de/book/show/luther_septembertestament_1522)
- Der automatisch erkannte Text aus Johann Crügers _Praxis Pietatis Melica_, basierend auf den vom MDZ München bereitgestellten OCR-Daten (https://www.digitale-sammlungen.de/en/view/bsb10589853)

Die Datensätze werden im Rahmen dieser Projekt unter einer Creative Commons 4.0 BY-NC-SA (Orgelpredigt-Datensätze, Altes Testament, Crüger)[https://creativecommons.org/licenses/by-nc-sa/4.0/], bzw. [Creative Commons 3.0 BY-SA](https://creativecommons.org/licenses/by-sa/3.0/) (Neues Testament) Lizenz bereitgestellt.

## Repositoriumsstruktur
Die Daten dieses Repositoriums sind wie folgt organisiert:
- core enthält interne Module für die Orgelpredigtaufarbeitung, sowie string- und vektorbasierte Ähnlichkeitssuche
- corpus_description enthält Notebooks zur statistischen Beschreibung des Orgelpredigtkorpus
- performance_tests enthält das Notebook zur automatischen Durchführung der Performancetests, sowie Ordner mit den Ergebnissen
- predictions enthält die für die Webdarstellung verwendeten automatisch generierten Listen mit textuellen Echos
- preprocessing enthält Notebooks anhand derer die Datensätze aus ihren Quelldateien aufbereitet wurden
- sermon_tables und sermons_chunked enthält die Orgelpredigten in Tabellen- bzw. JSON-Format für die Analyse aufbereitet
- similarity_searches enthält Notebooks mit testsetups für die Entwicklung der Performancetests, die in das in performance_tests enthaltene Notebook eingeflossen sind
- source_texts enthält die aufbereiteten Datensätze
- streamlit enthält die Skripte für die interaktive Webansicht

## Dank, Links und Referenzen
Diese Projekt wäre ohne die folgenden Resourcen nicht umsetzbar gewesen:
- Das DFG-Projekt _Deutsche Orgelpredigtdrucke zwischen 1600 und 1800 – Katalogisierung, Texterfassung, Auswertung_ (https://orgelpredigt.ur.de), unter Mitarbeit von Katelijne Schiltz, Lucinde Braun, u.a.
- Die Edition der Lutherübersetzung des Neuen Testaments (https://www.deutschestextarchiv.de/book/show/luther_septembertestament_1522) unter Mitarbeit von Magdalena Schulze, Benjamin Fiechter, u.a.
- Die Edition der Lutherübersetzung des Alten Testaments (https://sourceforge.net/projects/zefania-sharp/files/Bibles/GER/Lutherbibel/Luther%201545%20%28Letzte%20Hand%29/) unter Mitarbeit von H.J.H.
- Das LaBSE-Modell (https://www.kaggle.com/models/google/labse/tensorFlow2/labse/)