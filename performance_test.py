# %%
import json
import re
import orgelpredigt_analysis as oa
from rapidfuzz import fuzz
import pandas as pd
import statistics
import os
import io
import pprint

import datetime
from numpyencoder import NumpyEncoder

#%%
def flatten(xss):
    return [x for xs in xss for x in xs]

def is_consecutive(L):
    return all(n-i==L[0] for i,n in enumerate(L))

def is_equal(L):
    return all(n == L[0] for n in L)

def is_song_in_book(id):
    match = re.findall(r'E10[0-9]{4}', id)[0]
    
    with open('songs_to_pages_mapping.json') as f:
        songbook_pages = json.load(f)
    if songbook_pages[match]["pages"] == '':
        return False
    else:
        return True
    
def song_page(id): 
    match = re.findall(r'E10[0-9]{4}', id)[0]
    with open('songs_to_pages_mapping.json') as f:
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

def reconsider_match(sent, pages):
    highest_match = 0
    matches = {}
    for page in pages:
        with open(f"source_texts/praxis_pietatis_verses/{page}.json") as f:
            verses = json.load(f)
        
        for verse in verses:
            sim_score = fuzz.ratio(sent, verse)
            if sim_score > highest_match:
                highest_match = sim_score
                matches[sim_score] = [verse, page]

    if highest_match > 0:
        return [matches[highest_match], highest_match]
    else:
        return [["no match", 0], 0.0]
    
def add_inferred_matches(guessed_hits: pd.DataFrame, sermon: oa.Sermon) -> pd.DataFrame:
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
                    match, sim_score = reconsider_match(missing_sent, [pages[0], pages[1]])
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

def correct_inbetween_matches(df: pd.DataFrame) -> pd.DataFrame:
    for i in range(0, len(df) - 3):
        chunk = df.iloc[i:i+3]
        pages = chunk["Liederbuch"].to_list()
        pars = chunk["Paragraph"].to_list()
        sents = chunk["Satz"].to_list()
        if (all(x==pars[0] for x in pars) and not is_equal(pages)):   # abort if paragraphs change or pages are already the same
            if pages[0] == pages[2]:
                missing_sent = chunk["Predigt"][chunk.index[1]]
                print(missing_sent)
                match, sim_score = reconsider_match(missing_sent, [pages[0]])
                if sim_score > 60:
                    verse = match[0]
                    page = match[1]
                    new_data = [missing_sent, pars[1], sents[1], page, verse, float(f"{sim_score:.2f}"), False]
                    df.loc[(df['Paragraph'] == pars[1]) & (df["Satz"] == sents[1])] = new_data
                    #df.iloc[i] = new_data

    return df

#%%
relevant_page_texts = []
#for n in page_nrs:
for n in range(41, 1291):
    with open(f"source_texts/praxis_pietatis_verses/{n}.json") as f:
        page = json.load(f)
    page_info = {}
    page_info[n] = page
    relevant_page_texts.append(page_info)

#%%
table_names = os.listdir("similarity_tables")

print("Welche Resultate aus der folgenden Liste sollen verglichen werden?")
for i in table_names:
    print(f"{i}")
response = None
while response not in table_names:
    response = input("Bitte Dateinamen eingeben")
# Now response is either "yes" or "no"

# %%
with open(f"similarity_tables/{response}") as f:
    sim_table = json.load(f)

date = sim_table['date']
method = sim_table['method']
fuzziness = sim_table['fuzziness']

# %%
test_score = {}
test_score["type"] = method
test_score["fuzziness"] = fuzziness
test_score["date"] = date
test_score["sermons"] = []

for result in sim_table['results']:
    id = result[0]
    table = result[1]

    print(f"Starting with {id}")
    sermon = oa.Sermon(id)

    buffer = io.StringIO(table)

    guessed_hits = pd.read_csv(buffer)    # create dataframe

    guessed_hits = remove_duplicates(guessed_hits).reset_index(drop=True)

    guessed_hits = add_inferred_matches(guessed_hits, sermon)
    guessed_hits = correct_inbetween_matches(guessed_hits)
    
    guessed_hits.sort_values("Satz", ascending=True, inplace=True)
    guessed_hits.reset_index(drop=True)

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

    # analysis
    val_hits = len(known_hits)
    
    merged_df = pd.merge(guessed_hits, known_hits, on=['Paragraph', 'Satz'], how='left', indicator=True)
    hits_not_in_val = len(merged_df[merged_df['_merge'] == 'left_only'].drop('_merge', axis=1))
    
    agreed_hits = converged_df["in_page_list"].value_counts()[True]
    divergent_hits = len(converged_df) - agreed_hits
    missed_hits = len(known_hits) - (agreed_hits + divergent_hits)
    avg_certainty = guessed_hits["Ähnlichkeit"].mean()

    precision = agreed_hits / (agreed_hits + divergent_hits + hits_not_in_val)
    recall = agreed_hits / val_hits

    f1 = (2 * precision * recall) / (precision + recall)

    results = {}

    results["id"] = id
    results["agreed_hits"] = agreed_hits
    results["divergent_hits"] = divergent_hits
    results["new_hits"] = hits_not_in_val
    results["missed_hits"] = missed_hits
    results["avg_certainty"] = avg_certainty

    results["precision"] = precision
    results["recall"] = recall
    results["f1-score"] = f1

    test_score["sermons"].append(results)

all_precision = [x["precision"] for x in test_score["sermons"]]
all_recall = [x["recall"] for x in test_score["sermons"]]
all_f1 = [x["f1-score"] for x in test_score["sermons"]]
all_avg_cert = [x["avg_certainty"] for x in test_score["sermons"]]

test_score["overall_precision"] = statistics.mean(all_precision)
test_score["overall_recall"] = statistics.mean(all_recall)
test_score["overall_f1"] = statistics.mean(all_f1)
test_score["overall_certainty"] = statistics.mean(all_avg_cert)

# %%
with open("test_results.json", "r") as f:
    test_results = json.load(f)

dates = [x['date'] for x in test_results]

if date not in dates:
    test_results.append(test_score)
    
    with open("test_results.json", "w") as f:
        json.dump(test_results, f, ensure_ascii=False, cls=NumpyEncoder)