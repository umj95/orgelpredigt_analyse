# %%
import json
import re
import core.utils as oa
from rapidfuzz import fuzz
import pandas as pd
import statistics
import os
import io
import pprint

from numpyencoder import NumpyEncoder

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents.base import Document
from langchain_core.runnables import chain
from typing import List
import math
from sentence_transformers import SentenceTransformer

from pathlib import Path

# root directory path
ROOT = Path(__file__).resolve().parents[1]

#%%
def flatten(xss):
    return [x for xs in xss for x in xs]

def is_consecutive(L):
    return all(n-i==L[0] for i,n in enumerate(L))

def is_equal(L):
    return all(n == L[0] for n in L)

def is_song_in_book(id):
    match = re.findall(r'E10[0-9]{4}', id)[0]
    
    with open(ROOT / 'songs_to_pages_mapping.json') as f:
        songbook_pages = json.load(f)
    if songbook_pages[match]["pages"] == '':
        return False
    else:
        return True
    
def song_page(id): 
    match = re.findall(r'E10[0-9]{4}', id)[0]
    with open(ROOT / 'songs_to_pages_mapping.json') as f:
        songbook_pages = json.load(f)
    page = songbook_pages[match]["pages"]
    
    return [int(page) + 42, int(page) + 43, int(page) + 44]

def check_page_proxy(numbers):
  """
  Checks if a list of numbers are either all the same or have a maximum difference of 1 between any two numbers.

  Args:
    numbers: A list of numbers.

  Returns:
    True if the numbers meet the criteria, False otherwise.  Returns False if the list is empty.
  """

  if not numbers:
    return False  # Handle empty list case

  first_number = numbers[0]
  all_same = True
  max_diff_one = True

  for number in numbers:
    if number != first_number:
      all_same = False
    if abs(number - first_number) > 1:
      max_diff_one = False

  return all_same or max_diff_one

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicate matches for sentences in quote classification

    Args:
        df (pd.DataFrame): The Dataframe containing the sentence matches

    Returns:
        pd.DataFrame: The Dataframe with duplicates removed
    """
    def find_duplicate_satz(df):
        duplicate_indices = {}
        for par_satz, indices in df.groupby(['Paragraph', 'Satz']).groups.items():
            if len(indices) > 1:
                satz_id = str(par_satz[0]) +  "-" + str(par_satz[1])
                duplicate_indices[satz_id] = list(indices)  # Convert indices to a list
        return duplicate_indices

    for satz_id, indices in find_duplicate_satz(df.copy()).items():
        satz = [int(x) for x in satz_id.split("-")]
        if indices[0]-1 in df.index:
            check = df["Liederbuch"][indices[0]-1]
        else:
            check = df["Liederbuch"][indices[-1]+1]
        matches = df.query(f"Paragraph == {satz[0]} and Satz == {satz[1]} and Liederbuch == {check}").index
        if len(matches):
            match_index = matches[0]
            for i in indices:
                if i != match_index:
                    df.drop([i], inplace=True)

    df.sort_values(by=["Ähnlichkeit"], inplace=True)
    df.drop_duplicates(subset=['Paragraph','Satz'], keep='last', inplace=True)
    df.sort_index(inplace=True)
    df.reset_index(drop=True)

    return df

def reconsider_match(sent, pages, retriever):
    highest_match = 1
    matches = {}
    for page in pages:
        hits = retriever.invoke({"query": sent, "page": page})
        for hit in hits:
            if hit.metadata["score"] < highest_match:
                highest_match = hit.metadata["score"]
                matches[highest_match] = [hit.page_content, page]

    if highest_match < 1:
        return [matches[highest_match], highest_match]
    else:
        return [["no match", 0], 1]
    
def add_inferred_matches(guessed_hits: pd.DataFrame, sermon: oa.Sermon, retriever) -> pd.DataFrame:
    for n in range(3):
        additional_matches = []
        sent_add = lambda x : [x+2,x+3,x+4]
        for i in range(0, len(guessed_hits) - 2):
            chunk = guessed_hits.iloc[i:i+2]
            pages = chunk["Liederbuch"].to_list()
            pars = chunk["Paragraph"].to_list()
            sents = chunk["Satz"].to_list()
            if all(x==pars[0] for x in pars):   # abort if paragraphs change
                if sents[1] in sent_add(sents[0]):
                    missing_sent = " ".join(sermon.chunked[pars[0]][sents[0]+1]["words"])
                    match, sim_score = reconsider_match(missing_sent, [pages[0], pages[1]], retriever)
                    if match[0] != "no match":
                        verse = match[0]
                        page = match[1]
                        additional_matches.append([missing_sent, 
                                                pars[0],
                                                sents[0]+1, 
                                                page, 
                                                verse, 
                                                float(f"{sim_score:.2f}"), 
                                                False])
                    
        new_matches = pd.DataFrame(additional_matches, columns=["Predigt", "Paragraph", "Satz", 
                                                        "Liederbuch", "Liedvers", 
                                                        "Ähnlichkeit", "Dopplung"])

        guessed_hits = pd.concat([guessed_hits, new_matches])
        guessed_hits.sort_values(["Paragraph", "Satz"], ascending=True, inplace=True)
        guessed_hits.reset_index(drop=True)
    
    return guessed_hits

def correct_inbetween_matches(df: pd.DataFrame, retriever) -> pd.DataFrame:
    for i in range(0, len(df) - 3):
        chunk = df.iloc[i:i+3]
        pages = chunk["Liederbuch"].to_list()
        pars = chunk["Paragraph"].to_list()
        sents = chunk["Satz"].to_list()
        if (all(x==pars[0] for x in pars) and not is_equal(pages)):   # abort if paragraphs change or pages are already the same
            if pages[0] == pages[2]:
                missing_sent = chunk["Predigt"][chunk.index[1]]
                match, sim_score = reconsider_match(missing_sent, [pages[0]], retriever)
                if sim_score > 60:
                    verse = match[0]
                    page = match[1]
                    new_data = [missing_sent, pars[1], sents[1], page, verse, float(f"{sim_score:.2f}"), False]
                    df.loc[(df['Paragraph'] == pars[1]) & (df["Satz"] == sents[1])] = new_data
                    #df.iloc[i] = new_data

    return df

#%%
table_names = os.listdir(ROOT / "similarity_tables")

print("Welche Resultate aus der folgenden Liste sollen verglichen werden?")
for i in table_names:
    print(f"{i}")
response = None
while response not in table_names:
    response = input("Bitte Dateinamen eingeben")

# %%
with open(ROOT / f"similarity_tables/{response}") as f:
    sim_table = json.load(f)

date = sim_table['date']
corpus = sim_table['corpus']
method = sim_table['method']
model_name = sim_table['model']
fuzziness = sim_table['fuzziness']

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
directory = str(ROOT / "chroma/chroma_db_{model_name}")
vectordb = Chroma(persist_directory=directory)
@chain
def retriever(inputs: dict) -> tuple[Document]:
    query = inputs["query"]
    page = inputs["page"]
    docs, scores = zip(
        *vectordb.similarity_search_with_score(
            query,
            k=1,
            filter={"source": str(page)}
        )
    )
    for doc, score in zip(docs, scores):
        doc.metadata["score"] = score
    return docs

# %% 
test_score = {}
test_score["type"] = method
test_score["corpus"] = corpus
test_score["fuzziness"] = fuzziness
test_score["date"] = date
test_score["sermons"] = []

for result in sim_table['results']:
    id = result[0]
    all_sents = result[1]
    table = result[2]

    print(f"Starting with {id}")
    sermon = oa.Sermon(id)

    buffer = io.StringIO(table)

    guessed_hits = pd.read_csv(buffer)    # create dataframe

    #guessed_hits = remove_duplicates(guessed_hits).reset_index(drop=True)

    guessed_hits = add_inferred_matches(guessed_hits, sermon, retriever)
    guessed_hits = correct_inbetween_matches(guessed_hits, retriever)

    guessed_hits.sort_values("Satz", ascending=True, inplace=True)
    guessed_hits.reset_index(drop=True)

    predicted_true_negatives = all_sents - len(guessed_hits)

    # create validation set
    validation = []
    for i in range(len(sermon.chunked)):                # for each paragraph
        for j in range(len(sermon.chunked[i])):         # for each sentence
            if " musikwerk" in sermon.chunked[i][j]["types"]:
                line = " ".join(sermon.chunked[i][j]["words"])
                refs = ", ".join(set(flatten(sermon.chunked[i][j]["references"])))
                validation.append([line, i, j, refs])

    known_hits = pd.DataFrame(validation, columns=["Predigt", "Paragraph", "Satz", "Referenz"])
    known_hits = known_hits[known_hits['Referenz'].apply(is_song_in_book)]
    known_hits["Ref_Seite"] = known_hits['Referenz'].apply(song_page)

    converged_df = pd.merge(known_hits, guessed_hits, on=['Paragraph','Satz'], how='inner')
    converged_df["in_page_list"]  = converged_df.apply(lambda row: row['Liederbuch'] in row['Ref_Seite'], axis=1)

    # analysis per verse
    val_hits_verse = len(known_hits)
    confirmed_true_negatives = all_sents - val_hits_verse
    
    merged_df = pd.merge(guessed_hits, known_hits, on=['Paragraph', 'Satz'], how='left', indicator=True)
    hits_not_in_val_verse = len(merged_df[merged_df['_merge'] == 'left_only'].drop('_merge', axis=1))
    
    agreed_hits_verse = converged_df["in_page_list"].value_counts()[True] # true pos
    divergent_hits_verse = len(converged_df) - agreed_hits_verse    # false pos
    missed_hits_verse = len(known_hits) - (agreed_hits_verse + divergent_hits_verse) # false neg
    avg_certainty = guessed_hits["Ähnlichkeit"].mean()

    true_negatives = predicted_true_negatives 

    tp = agreed_hits_verse
    tn = true_negatives
    fp = hits_not_in_val_verse
    fn = missed_hits_verse

    precision_verse = tp / (tp + fp)
    recall_verse = agreed_hits_verse / (tp + fn)

    f1_verse = (2 * precision_verse * recall_verse) / (precision_verse + recall_verse)

    accuracy_verse = (agreed_hits_verse + (all_sents - (agreed_hits_verse + divergent_hits_verse + missed_hits_verse))) / all_sents

    mc_verse = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    # analysis per hit
    grouped_classifications = guessed_hits.copy().groupby(["Paragraph", "Liederbuch"])
    nr_of_classifications = len(list(grouped_classifications.groups.keys()))

    grouped_known_hits = known_hits.copy().groupby(["Referenz", "Paragraph"])
    val_hits = len(list(grouped_known_hits.groups.keys()))

    new_hits = merged_df[merged_df['_merge'] == 'left_only'].drop('_merge', axis=1)
    grouped_new_hits = new_hits.copy().groupby(["Paragraph", "Liederbuch"])
    hits_not_in_val = len(list(grouped_new_hits.groups.keys()))

    grouped_hits = converged_df.copy().groupby(["Referenz", "Paragraph"])
    group_keys = list(grouped_hits.groups.keys())

    page_matches = 0
    page_mismatches = 0

    for name, group in grouped_hits:
        known_pages = group['Ref_Seite'].iloc[0]
        guessed_pages = group["Liederbuch"].to_list()
        
        if len(set(known_pages).intersection(guessed_pages)) > 0:
            page_matches += 1
        else: 
            page_mismatches += 1
    
    agreed_hits = page_matches
    divergent_hits = page_mismatches
    missed_hits = val_hits - (agreed_hits + divergent_hits)

    precision_hits = agreed_hits / (agreed_hits + divergent_hits + hits_not_in_val)
    recall_hits = agreed_hits / val_hits

    f1_hits = (2 * precision_hits * recall_hits) / (precision_hits + recall_hits)

    results = {}

    results["id"] = id
    results["identified_hits_total"] = nr_of_classifications
    results["song_quotes_total"] = val_hits
    results["sentences total"] = all_sents
    results["verse_agreed_hits"] = agreed_hits_verse
    results["verse_divergent_hits"] = divergent_hits_verse
    results["verse_new_hits"] = hits_not_in_val_verse
    results["verse_missed_hits"] = missed_hits_verse
    results["verse_avg_certainty"] = avg_certainty
    results["verse_matthews_coeff"] = mc_verse

    results["verse_precision"] = precision_verse
    results["verse_recall"] = recall_verse
    results["verse_f1-score"] = f1_verse
    results["verse_accuracy"] = accuracy_verse

    results["hits_agreed"] = agreed_hits
    results["hits_divergent"] = divergent_hits
    results["hits_new"] = hits_not_in_val
    results["hits_missed"] = missed_hits

    results["hits_precision"] = precision_hits
    results["hits_recall"] = recall_hits
    results["hits_f1-score"] = f1_hits

    test_score["sermons"].append(results)

all_precision_verse = [x["verse_precision"] for x in test_score["sermons"]]
all_recall_verse = [x["verse_recall"] for x in test_score["sermons"]]
all_f1_verse = [x["verse_f1-score"] for x in test_score["sermons"]]
all_accuracy_verse = [x["verse_accuracy"] for x in test_score["sermons"]]
all_avg_cert = [x["verse_avg_certainty"] for x in test_score["sermons"]]
all_mattews_coeff = [x["verse_matthews_coeff"] for x in test_score["sermons"]]

all_precision_hits = [x["hits_precision"] for x in test_score["sermons"]]
all_recall_hits = [x["hits_recall"] for x in test_score["sermons"]]
all_f1_hits = [x["hits_f1-score"] for x in test_score["sermons"]]

test_score["overall_precision_verse"] = statistics.mean(all_precision_verse)
test_score["overall_recall_verse"] = statistics.mean(all_recall_verse)
test_score["overall_f1_verse"] = statistics.mean(all_f1_verse)
test_score["overall_certainty_verse"] = statistics.mean(all_avg_cert)
test_score["overall_accuracy_verse"] = statistics.mean(all_accuracy_verse)
test_score["overall_matthews_coeff_verse"] = statistics.mean(all_mattews_coeff)

test_score["overall_precision_hits"] = statistics.mean(all_precision_hits)
test_score["overall_recall_hits"] = statistics.mean(all_recall_hits)
test_score["overall_f1_hits"] = statistics.mean(all_f1_hits)

# %%
pprint.pprint(test_score)
# %%
with open(ROOT / f"test_results_{corpus}.json", "r") as f:
    test_results = json.load(f)

dates = [x['date'] for x in test_results]

if date not in dates:
    test_results.append(test_score)

    with open(ROOT / f"test_results_{corpus}.json", "w") as f:
        json.dump(test_results, f, ensure_ascii=False, cls=NumpyEncoder)

# %%
