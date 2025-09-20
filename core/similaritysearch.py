
import re
import core.utils as oa
from rapidfuzz import fuzz
import pandas as pd
import os
import json

from numpyencoder import NumpyEncoder
from pathlib import Path

# root directory path
ROOT = Path(os.getcwd()).resolve().parents[0]

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

def reconsider_match(sent, pages):
    highest_match = 0
    matches = {}
    if type(pages[0]) == int:
        for page in pages:
            with open(ROOT / f"source_texts/praxis_pietatis_verses/{page}.json") as f:
                verses = json.load(f)
            
            for verse in verses:
                sim_score = fuzz.ratio(sent, verse)
                if sim_score > highest_match:
                    highest_match = sim_score
                    matches[sim_score] = [verse, page]
    else:
        for page in pages:
            book, chap, vers = page.split("_")
            with open(ROOT / f"source_texts/bible/old_testament_chunked.json") as f:
                bible = json.load(f)
            with open(ROOT / f"source_texts/bible/new_testament_chunked.json") as f:
                bible.update(json.load(f))
            verses = []
            for key in bible[book][chap]:
                verses.extend(bible[book][chap][key])

            for verse in verses:
                sim_score = fuzz.ratio(sent, verse)
                if sim_score > highest_match:
                    highest_match = sim_score
                    matches[sim_score] = [verse, page]

    if highest_match > 0:
        return [matches[highest_match], highest_match]
    else:
        return [["no match", 0], 0.0]
    
def add_inferred_matches(guessed_hits: pd.DataFrame, id: str) -> pd.DataFrame:
    sermon = oa.Sermon(id)
    for n in range(3):
        additional_matches = []
        sent_add = lambda x : [x+2,x+3,x+4]
        for i in range(0, len(guessed_hits) - 2):
            chunk = guessed_hits.iloc[i:i+2]
            pages = chunk["Fundstelle"].to_list()
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
                                                        "Fundstelle", "Vers", 
                                                        "Ähnlichkeit", "Dopplung"])

        guessed_hits = pd.concat([guessed_hits, new_matches])
        guessed_hits.sort_values(["Paragraph", "Satz"], ascending=True, inplace=True)
        guessed_hits.reset_index(drop=True)
    
    return guessed_hits

def correct_inbetween_matches(df: pd.DataFrame) -> pd.DataFrame:
    for i in range(0, len(df) - 3):
        chunk = df.iloc[i:i+3]
        pages = chunk["Fundstelle"].to_list()
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
            check = df["Fundstelle"][indices[0]-1]
        else:
            check = df["Fundstelle"][indices[-1]+1]
        matches = df.query(f"Paragraph == {satz[0]} and Satz == {satz[1]} and Fundstelle == '{check}'").index
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


def find_similarities(task: str, id: str, relevant_texts: list, fuzziness: int, test=False) -> pd.DataFrame:
    """Find passages from a list of texts in a list of sermons  

    Args:
        task (str): "lieder", or "bibel"
        id (str): the id of the sermon in which similarities are to be searched
        relevant_texts (list): list of texts in which similarities are to be found
        fuzziness (int): The fuzziness factor
        test (bool): If set to True words marked "bible" are filtered for "lieder" and vice versa

    Returns:
        pd.Dataframe: a dataframe with found passages
    """
    if task == "lieder":
        print(f"Starting with {id}")
        sermon = oa.Sermon(id)

        # perform classification
        hits = []
        all_sents = 0
        for i in range(len(sermon.chunked)):                # for each paragraph
            for j in range(len(sermon.chunked[i])):         # for each sentence
                if " bibel" in sermon.chunked[i][j]["types"] and test:
                    continue
                else:
                    all_sents += 1
                    query = " ".join(sermon.chunked[i][j]["words"])
                    query = re.sub(r'[/.,;:?!]', '', query)
                    for page in relevant_texts:
                        for pagenr, verses in page.items():
                            for verse in verses:
                                sim_score = fuzz.ratio(query, verse)
                                if sim_score >= fuzziness:
                                    hits.append([query, i, j, pagenr, verse, float(f"{sim_score:.2f}")])

        guessed_hits = pd.DataFrame(hits, columns=["Predigt", "Paragraph", "Satz", "Fundstelle", "Vers", "Ähnlichkeit"])     # create dataframe
        guessed_hits['Dopplung'] = guessed_hits.groupby('Satz')['Satz'].transform(lambda x: x.duplicated())

        guessed_hits = remove_duplicates(guessed_hits).reset_index(drop=True)

    else:
        print(f"Starting with {id}")
        sermon = oa.Sermon(id)

        # perform classification
        hits = []
        all_sents = 0
        for i in range(len(sermon.chunked)):                # for each paragraph
            for j in range(len(sermon.chunked[i])):         # for each sentence
                if " musikwerk" in sermon.chunked[i][j]["types"] and test:
                    continue
                else:
                    all_sents += 1
                    query = " ".join(sermon.chunked[i][j]["words"])
                    query = re.sub(r'[/.,;:?!]', '', query)
                    for page in relevant_texts:
                        for pagenr, verses in page.items():
                            for verse in verses:
                                sim_score = fuzz.ratio(query, verse)
                                if sim_score >= fuzziness:
                                    hits.append([query, i, j, pagenr, verse, float(f"{sim_score:.2f}")])

        guessed_hits = pd.DataFrame(hits, columns=["Predigt", "Paragraph", "Satz", "Fundstelle", "Vers", "Ähnlichkeit"])     # create dataframe
        guessed_hits['Dopplung'] = guessed_hits.groupby('Satz')['Satz'].transform(lambda x: x.duplicated())

        guessed_hits = remove_duplicates(guessed_hits).reset_index(drop=True)
    
    return guessed_hits
