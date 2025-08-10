import db_connection
from mysql.connector import Error
import pandas as pd
import ast
import re
import json

cursor, connection = db_connection.get_connection()

def is_id(value: str) -> bool:
    pattern = re.compile(r'E[01][0-9]{5}')
    if re.match(pattern, value):
        return True
    else:
        return False

def get_short_info(id: str) -> str:
    if id.startswith("E00"):
        x = Orgelpredigt(id)
        return f"{x.autor}: {x.kurztitel}"
    elif id.startswith("E01"):
        x = Person(id)
        return f"{x.name} ({x.daten})"
    elif id.startswith("E03"):
        x = Place(id)
        return f"{x.name}"
    elif id.startswith(("E08", "E09")):
        x = Source(id)
        return f"{x.autor}: {x.titel} ({x.jahr})"
    elif id.startswith("E10"):
        x = Musikwerk(id)
        return f"{x.komponist}: {x.titel}"
    else:
        return "Ungültige ID"
    
class Place:
    def __init__(self, id, conn=connection):
        self.id = id
        self.cursor = conn.cursor()
        if is_id(id):
            self.get_place_info()
        else:
            self.name = id
            self.gnd = ""
            self.typ = ""
            self.gebiet = ""
            self.koordinaten = ""

    def __str__(self):
        return f"{self.name}"

    def get_place_info(self):
        try:
            self.cursor.execute(f"SELECT e03id, e03name, e03gnd, e03typ, e03gebiet, e03koordinaten FROM e03_geographica WHERE e03id = '{self.id}'")
            results = self.cursor.fetchall()

            if results:
                column_names = [col[0] for col in self.cursor.description]
                data = [dict(zip(column_names, row))  
                    for row in results][0]
        
                self.name = data["e03name"]
                self.gnd = data["e03gnd"]
                self.typ = data["e03typ"]
                self.gebiet = data["e03gebiet"]
                self.koordinaten = data["e03koordinaten"]
            else:
                print(f"Query executed for {self.id}, but no data found.")
                self.name = "no_name"
                self.gnd = ""
                self.typ = ""
                self.gebiet = ""
                self.koordinaten = ""
        
        except Error as e:
            print(f"Database error occurred for {self.id}:", e)
        except Exception as e:
            print(f"Unexpected error for {self.id}:", e)

class Person:
    def __init__(self, id, conn=connection):
        self.id = id
        self.cursor = conn.cursor()
        if is_id(id):
            self.get_person_info()
        else:
            self.name = id
            self.wirkungsorte = ""
            self.geburtsort = ""
            self.sterbeort = ""

    def __str__(self):
        if is_id(self.id):
            return f"{self.name}, {self.geburtsdatum} ({self.geburtsort})-{self.sterbedatum} ({self.sterbeort})"
        else:
            return f"{self.name}"

    def get_person_info(self):
        try:
            self.cursor.execute(f"SELECT  e01id, e01gnd, e01nachname, e01vorname, e01akademisch, e01geburtsdatum, e01geburtsort, e01sterbedatum, e01sterbeort, e01rahmendaten, e01daten, e01wirkungsorte, e01funktionen FROM e01_personen WHERE e01id = '{self.id}'")
            column_names = [col[0] for col in self.cursor.description]
            results = self.cursor.fetchall()
            if results:
                data = [dict(zip(column_names, row))  
                    for row in results][0]
                
                self.name = " ".join([data["e01vorname"], data["e01nachname"]])
                self.vorname = data["e01vorname"]
                self.nachname = data["e01nachname"]
                self.gnd = data["e01gnd"]
                self.akademisch = data["e01akademisch"]
                self.geburtsdatum = data["e01geburtsdatum"]
                self.sterbedatum = data["e01sterbedatum"]
                self.daten = data["e01rahmendaten"]
                self.wirkungsorte = data["e01wirkungsorte"]
                self.funktionen = data["e01funktionen"]
                
                self.geburtsort = Place(data["e01geburtsort"])
                self.sterbeort = Place(data["e01sterbeort"])
            else:
                print(f"Query executed for {self.id}, but no data found.")
                self.name = "no_name"
                self.wirkungsorte = ""
                self.geburtsort = ""
                self.sterbeort = ""

        except Error as e:
            print(f"Database error occurred for {self.id}:", e)
        except Exception as e:
            print(f"Unexpected error for {self.id}:", e)

    
    def parse_wirkungsorte(self):
        wirkungsorte = []
        stops = [x.strip() for x in self.wirkungsorte.split(";")]
        for stop in stops:
            location = re.findall(r'E03[0-9]{4}', stop)
            if location:
                place = Place(location[0])
                item = re.sub(r"E03[0-9]{4}", str(place), stop)
                wirkungsorte.append({"item": item, "koordinaten": place.koordinaten})
            else:
                wirkungsorte.append({"item": stop, "koordinaten": ""})
        
        return wirkungsorte

    def get_personal_network(self):
        network = {
            "Geburtsort": self.geburtsort.koordinaten,
            "Sterbeort": self.sterbeort.koordinaten
        }
        
        for i in self.parse_wirkungsorte():
            network[i["item"]] = i["koordinaten"]
        
        return network
    
class Source:
    def __init__(self, id, conn=connection):
        self.id = id
        self.cursor = conn.cursor()
        if is_id(id):
            self.get_quelle_info()
            self.get_literatur_info()
        else:
            self.autor = id
            self.titel = id
            self.ort = ""
            self.jahr = ""
    
    def __str__(self):
        return f"{self.autor}: {self.titel} ({self.ort}: {self.jahr})"

    def get_quelle_info(self):
        if self.id.startswith("E08"):
            try:
                self.cursor.execute(f"SELECT e08id, e08autor1, e08titel1, e08band1, e08ort, e08jahr, e08verlag, e08typ, e08vdnummer, e08dnbnummer FROM e08_quellen WHERE e08id = '{self.id}'")
                column_names = [col[0] for col in self.cursor.description]
                results = self.cursor.fetchall()
                if results:
                    data = [dict(zip(column_names, row))  
                        for row in results][0]
                    self.autor = data["e08autor1"]
                    self.titel = data["e08titel1"]
                    self.band = data["e08band1"]
                    self.ort = data["e08ort"]
                    self.jahr = data["e08jahr"]
                    self.verlag = data["e08verlag"]
                    self.typ = data["e08typ"]
                    self.vdnummer = data["e08vdnummer"]
                    self.dnbnummer = data["e08dnbnummer"]
                else:
                    print(f"Query executed for {self.id}, but no data found.")
                    self.autor = "no_author"
                    self.titel = "no_title"
                    self.ort = "no_place"
                    self.jahr = "no_year"
            except Error as e:
                print(f"Database error occurred for {self.id}:", e)
            except Exception as e:
                print(f"Unexpected error for {self.id}:", e)
                    
    
    def get_literatur_info(self):
        if self.id.startswith("E09"):
            try:
                self.cursor.execute(f"SELECT e09id, e09autor1, e09titel1, e09band1, e09ort, e09jahr, e09verlag, e09typ, e09dnbnummer FROM e09_literatur WHERE e09id = '{self.id}'")
                column_names = [col[0] for col in self.cursor.description]
                results = self.cursor.fetchall()
                if results:
                    data = [dict(zip(column_names, row))  
                        for row in results][0]
                    self.autor = data["e09autor1"]
                    self.titel = data["e09titel1"]
                    self.band = data["e09band1"]
                    self.ort = data["e09ort"]
                    self.jahr = data["e09jahr"]
                    self.verlag = data["e09verlag"]
                    self.typ = data["e09typ"]
                    self.dnbnummer = data["e09dnbnummer"]
                else:
                    print(f"Query executed for {self.id}, but no data found.")
                    self.autor = "no_author"
                    self.titel = "no_title"
                    self.ort = "no_place"
                    self.jahr = "no_year"
            except Error as e:
                print(f"Database error occurred for {self.id}:", e)
            except Exception as e:
                print(f"Unexpected error for {self.id}:", e)

class Musikwerk:
    def __init__(self, id, conn=connection):
        self.id = id
        self.cursor = conn.cursor()
        if is_id(id):
            self.get_musikwerk_info()
        else:
            self.komponist = id
            self.titel = id
            self.kurztitel = ""
            self.gattung = ""
            self.besetzung = ""
    
    def __str__(self):
        return f"{self.komponist}: {self.titel}"

    def get_musikwerk_info(self):
        if self.id.startswith("E10"):
            try:
                self.cursor.execute(f"SELECT e10id, e10komponist, e10werk, e10kurztitel, e10textdichter, e10gattung, e10besetzung FROM e10_musikwerke WHERE e10id = '{self.id}'")
                column_names = [col[0] for col in self.cursor.description]
                results = self.cursor.fetchall()
                if results:
                    data = [dict(zip(column_names, row))  
                        for row in results][0]
                    self.komponist = data["e10komponist"]
                    self.titel = data["e10werk"]
                    self.kurztitel = data["e10kurztitel"]
                    self.gattung = data["e10gattung"]
                    self.besetzung = data["e10besetzung"]
                    #self.ort = data["e10ort"]
                    #self.jahr = data["e10jahr"]
                    #self.verlag = data["e10verlag"]
                else:
                    print(f"Query executed for {self.id}, but no data found.")
                    self.komponist = "no_composer"
                    self.titel = "no_title"
                    self.kurztitel = ""
                    self.gattung = ""
                    self.besetzung = ""
            except Error as e:
                print(f"Database error occurred for {self.id}:", e)
            except Exception as e:
                print(f"Unexpected error for {self.id}:", e)

class Orgelpredigt:
    def __init__(self, id, conn=connection):
        self.id = id
        self.cursor = conn.cursor()
        if is_id(id):
            self.get_orgelpredigt_info()
        else:
            self.autor.nachname = "--"
            self.autor.vorname = "--"
            self.kurztitel = id

    def __str__(self):
        return f"{self.autor.nachname}, {self.autor.vorname}: {self.kurztitel}"

    def get_orgelpredigt_info(self):
        try:
            self.cursor.execute(f"SELECT e00autor, e00kurztitel FROM e00_orgelpredigten WHERE e00id = '{self.id}'")
            column_names = [col[0] for col in self.cursor.description]
            results = self.cursor.fetchall()
            if results:
                sermon_info = [dict(zip(column_names, row)) for row in results][0]
                self.autor = Person(sermon_info["e00autor"])
                self.kurztitel = sermon_info["e00kurztitel"]
            else: 
                print(f"Query executed for {self.id}, but no data found.")
                self.autor.nachname = "--"
                self.autor.vorname = "--"
                self.kurztitel = self.id
        except Error as e:
            print(f"Database error occurred for {self.id}:", e)
        except Exception as e:
            print(f"Unexpected error for {self.id}:", e)

class Sermon:
    def __init__(self, id, conn=connection):
        self.id = id
        self.cursor = conn.cursor()
        self.get_sermon_info()
        self.get_sermon_table()
        self.get_quotations()
        self.get_sermon_chunked()
    
    def __str__(self):
        return f"{self.autor.nachname}, {self.autor.vorname}: {self.kurztitel}"
    
    def get_sermon_info(self):
        try:
            self.cursor.execute(f"SELECT e00autor, e00kurztitel, e00volltitel, e00verlagsort, e00verleger, e00jahr, e00umfang, e00konfession, e00bibelstelle, e00sonntag, e00einweihungsort FROM e00_orgelpredigten WHERE e00id = '{self.id}'")
            column_names = [col[0] for col in self.cursor.description]
            results = self.cursor.fetchall()
            if results:
                sermon_info = [dict(zip(column_names, row)) for row in results][0]
                self.kurztitel = sermon_info["e00kurztitel"]
                self.volltitel = sermon_info["e00volltitel"]
                self.erscheinungsjahr = sermon_info["e00jahr"]
                self.umfang = sermon_info["e00umfang"]
                self.konfession = sermon_info["e00konfession"]
                self.bibelstelle = sermon_info["e00bibelstelle"]
                self.sonntag = sermon_info["e00sonntag"]
                self.einweihungsort = Place(sermon_info["e00einweihungsort"])
                self.verlagsort = Place(sermon_info["e00verlagsort"])
                self.autor = Person(sermon_info["e00autor"])
                self.verleger = Person(sermon_info["e00verleger"])
            else: 
                print(f"Query executed for {self.id}, but no data found.")
        except Error as e:
            print(f"Database error occurred for {self.id}:", e)
        except Exception as e:
            print(f"Unexpected error for {self.id}:", e)
            

    def get_sermon_table(self):
        df = pd.read_csv(f'sermon_tables/{self.id}.tsv', sep='\t')
        self.words = df["word"].tolist()                                    # a list of all words
        self.word_types = df["types"].tolist()                              # a list of all word types
        self.reference = df["reference"].apply(ast.literal_eval).tolist()   # a list of all ids of references
        self.all_references = sum(self.reference, [])

    def get_sermon_chunked(self):
        with open(f"sermons_chunked/{self.id}.json", "r") as f:
            sermon_chunked = json.load(f)
        self.chunked = sermon_chunked

    
    def get_quotations(self):
        quoted_source_ids = []
        quoted_sources = []
        quoted_bible = []
        quoted_bible_verse = []
        quoted_music = []
        quoted_orgelpredigt = []
        source_id = re.compile(r"E[01][0-9]{5}")

        unique_refs = set([x for xs in self.reference for x in xs])
        for i in unique_refs:
            if i:
                if re.match(source_id, i):
                    quoted_source_ids.append(i)
                else:
                    quoted_bible.append(i)
        
        for i in quoted_source_ids:
            hits = self.all_references.count(i)
            if i.startswith("E00"):
                quoted_orgelpredigt.append({"item": Orgelpredigt(i), "word_share": hits})
            elif i.startswith("E08"):
                quoted_sources.append({"item": Source(i), "word_share": hits})
            elif i.startswith("E09"):
                quoted_sources.append({"item": Source(i), "word_share": hits})
            elif i.startswith("E10"):
                quoted_music.append({"item": Musikwerk(i), "word_share": hits})
        
        for i in quoted_bible:
            hits = self.all_references.count(i)
            quoted_bible_verse.append({"item": i, "word_share": hits})

        self.bibelzitate = quoted_bible_verse
        self.literaturzitate = quoted_sources
        self.orgelpredigtzitate = quoted_orgelpredigt
        self.musikzitate = quoted_music
