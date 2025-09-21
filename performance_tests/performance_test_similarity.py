# %%
import json
import re
import core.utils as oa
from rapidfuzz import fuzz
import pandas as pd
import statistics
import os
import io
from pprint import pprint
import math 

import datetime
from numpyencoder import NumpyEncoder

from pathlib import Path
import core.similaritysearch as simsearch

# root directory path
ROOT = Path(__file__).resolve().parents[1]

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
fuzziness = sim_table['fuzziness']

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

    guessed_hits = simsearch.remove_duplicates(guessed_hits).reset_index(drop=True)

    guessed_hits = simsearch.add_inferred_matches(guessed_hits, sermon.id)
    guessed_hits = simsearch.correct_inbetween_matches(guessed_hits)
    
    guessed_hits.sort_values("Satz", ascending=True, inplace=True)
    guessed_hits.reset_index(drop=True)

    predicted_true_negatives = all_sents - len(guessed_hits)

    # create validation set
    validation = []
    for i in range(len(sermon.chunked)):                # for each paragraph
        for j in range(len(sermon.chunked[i])):         # for each sentence
            if " musikwerk" in sermon.chunked[i][j]["types"]:
                line = " ".join(sermon.chunked[i][j]["words"])
                refs = ", ".join(set(simsearch.flatten(sermon.chunked[i][j]["references"])))
                validation.append([line, i, j, refs])

    known_hits = pd.DataFrame(validation, columns=["Predigt", "Paragraph", "Satz", "Referenz"])
    known_hits = known_hits[known_hits['Referenz'].apply(simsearch.is_song_in_book)]
    known_hits["Ref_Seite"] = known_hits['Referenz'].apply(simsearch.song_page)

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
pprint(test_score)
# %%
with open(ROOT / f"test_results_{corpus}.json", "r") as f:
    test_results = json.load(f)

dates = [x['date'] for x in test_results]

if date not in dates:
    test_results.append(test_score)

    with open(ROOT / f"test_results_{corpus}.json", "w") as f:
        json.dump(test_results, f, ensure_ascii=False, cls=NumpyEncoder)

# %%
guessed_hits['Paragraph'] = guessed_hits['Paragraph'].apply(lambda x: int(x))
guessed_hits['Satz'] = guessed_hits['Satz'].apply(lambda x: int(x))
guessed_hits = guessed_hits.sort_values(['Paragraph', 'Satz'])

