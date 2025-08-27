# %%
from typing import Iterable, List
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents.base import Document
from langchain_core.runnables import chain
from pprint import pprint
import json
import orgelpredigt_analysis as oa
import re
import pandas as pd
import datetime
from functools import reduce
from sentence_transformers import SentenceTransformer

import nltk
from nltk.corpus import stopwords

# %%
nltk.download('stopwords')
nltk.download('german')

german_stop_words = set(stopwords.words('german'))

orgel_stop_words = {'herr', 'gott', 'gottes', 'jesus', 'jesu', 'christus', 'christi', 'christe', 'christen', 'amen', 'heilig', 'heiliger', 'geist', 'sohn'}

stop_words = german_stop_words.union(orgel_stop_words)

# %%
def flatten_reduce(matrix):
    return list(reduce(lambda x, y: x + y, matrix, []))

def is_equal(L):
    return all(n == L[0] for n in L)

def find_shared_nums(set1, set2, set3):
    common_numbers = set()

    common12 = set1.intersection(set2)
    common_numbers.update(common12)

    common13 = set1.intersection(set3)
    common_numbers.update(common13)

    common23 = set2.intersection(set3)
    common_numbers.update(common23)

    return common_numbers

def remove_duplicates(df):
    grouped_hits = df.copy().groupby(["Paragraph", "Satz"])

    group_keys = list(grouped_hits.groups.keys())

    page_add = lambda x : [x, x+1,x+2]
    sent_add = lambda x : [x, x+1, x+2, x+3]

    for i in range(0, len(group_keys)-2):
        keys = group_keys[i:i+3]
        all_pages = []
        if is_equal([x[0] for x in keys]):
            for key in keys:
                current_group = grouped_hits.get_group(key)
                group_pages = current_group["Liederbuch"].to_list()
                potential_pages = flatten_reduce([page_add(x) for x in group_pages])
                all_pages.append(set(potential_pages))
            shared_page = find_shared_nums(all_pages[0], all_pages[1], all_pages[2])
            if len(shared_page) > 0:
                all_right_pages = []
                all_wrong_pages = []
                for key in keys:
                    current_group = grouped_hits.get_group(key)
                    if bool(len(current_group[current_group["Liederbuch"].isin(shared_page)])):
                        right_pages = current_group[current_group["Liederbuch"].isin(shared_page)]
                        wrong_pages = current_group[~current_group["Liederbuch"].isin(shared_page)]
                        all_wrong_pages += wrong_pages.index.to_list()
                        all_right_pages += right_pages.index.to_list()
                for idx in all_wrong_pages:
                    if idx in df.index:
                        df.drop(idx, inplace=True)
                for idx in all_right_pages:
                    if idx in df.index:
                        df.at[idx, 'Validated'] = True

    df.sort_values(by=["Ähnlichkeit"], inplace=True)
    df.drop_duplicates(subset=['Paragraph','Satz'], keep='last', inplace=True)
    df.sort_index(inplace=True)

    return df


# %%
#embeddings = OllamaEmbeddings(
#    model=model
#)
model_name = "LaBSE"
model = SentenceTransformer(f'sentence-transformers/{model_name}')

from langchain_core.embeddings.embeddings import Embeddings
class EmbedSomething(Embeddings):
    def __init__(self,model) -> None:
        self.model = model

    def embed_documents(self,texts):
        t = self.model.encode(texts)
        return t.tolist()

    def embed_query(self, text: str) -> List[float]:
        t = self.model.encode(text)
        return t.tolist()


emb = EmbedSomething(model)

#%%
vectordb = Chroma(persist_directory=f"./chroma/chroma_db_{model_name}", embedding_function=emb)

@chain
def retriever(query: str) -> tuple[Document]:
    docs, scores = zip(*vectordb.similarity_search_with_score(query, k=4))
    for doc, score in zip(docs, scores):
        doc.metadata["score"] = score

    return docs

# %%

# %%
print("Sollen Predigten mit den meisten (1), oder den längsten (2) Liedzitaten analysiert werden?")
response = None
while response not in ["1", "2"]:
    response = input("Bitte Dateinamen eingeben")

if response == "2":
    corpus = "longest"
else:
    corpus = "most"

with open(f"sermons_with_{corpus}_music.json", "r") as f:
    testsermons = json.load(f)

# %%
cosine_cutoff = 0.3
date = datetime.datetime.now().strftime("%y-%m-%d_%H:%M")

# %%
similarity_table = {}
similarity_table['date'] = date
similarity_table["corpus"] = corpus
similarity_table['method'] = 'vector_database'
similarity_table['model'] = model_name
similarity_table['fuzziness'] = cosine_cutoff
similarity_table['comments'] = "LaBSE-Modell mit Stopwörtern"

similarity_table['results'] = []

for id in testsermons:
    print(f"starting with {id}")
    hits = []
    sermon = oa.Sermon(id)
    sent_nr = 0
    for i in range(len(sermon.chunked)):                # for each paragraph
            for j in range(len(sermon.chunked[i])):         # for each sentence
                if " bibel" in sermon.chunked[i][j]["types"]:
                    continue
                else:
                    sent_nr += 1
                    words = sermon.chunked[i][j]["words"]
                    filtered_words = words
                    #filtered_words = [word for word in words if word.lower() not in stop_words]
                    query = " ".join(filtered_words)
                    query = re.sub(r'[/.,;:?!]', '', query)
                    matches = retriever.invoke(query)
                    for match in matches:
                        if match.metadata["score"] < cosine_cutoff:
                            hits.append([query, i, j, match.page_content, int(match.metadata["source"]), match.metadata["score"], False])
    
    guessed_hits = pd.DataFrame(hits, columns=["Predigt", "Paragraph", "Satz", "Liedvers", "Liederbuch", "Ähnlichkeit", "Validated"])     # create dataframe
    guessed_hits['Dopplung'] = guessed_hits.groupby('Satz')['Satz'].transform(lambda x: x.duplicated())

    guessed_hits = remove_duplicates(guessed_hits)

    guessed_hits =guessed_hits.drop(guessed_hits[(guessed_hits['Validated'] == False) & 
                               (guessed_hits['Ähnlichkeit'] > 0.2)].index)

    similarity_table['results'].append([id, sent_nr, guessed_hits.to_csv()[1:]])

# %%
with open(f'similarity_tables/vector_search_{corpus}_{cosine_cutoff}_{date}.json', "w") as f:
    json.dump(similarity_table, f, ensure_ascii=False)

# %%
guessed_hits
