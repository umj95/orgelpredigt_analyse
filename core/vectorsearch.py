# %%
import json
import core.utils as oa
import re
import pandas as pd
import datetime
from functools import reduce

from nltk.corpus import stopwords
from pathlib import Path

# root directory path
ROOT = Path(__file__).resolve().parents[1]

# %%
#nltk.download('stopwords')
#nltk.download('german')

german_stop_words = set(stopwords.words('german'))
em_stopwords = []
for stopword in german_stop_words:
    if "i" in stopword:
        em_stopwords.append(stopword)
        em_stopwords.append(stopword.replace("i", "j"))
    elif "u" in stopword:
        em_stopwords.append(stopword)
        em_stopwords.append(stopword.replace("u", "v"))
    else:
        em_stopwords.append(stopword)

em_stopwords = set(em_stopwords)

orgel_stop_words = {'herr', 'gott', 'gottes', 'jesus', 'jesu', 'christus', 'christi', 'christe', 'christen', 'amen', 'heilig', 'heiliger', 'geist', 'sohn'}

#stop_words = german_stop_words.union(orgel_stop_words)

# %%
def flatten(xss):
    return [x for xs in xss for x in xs]

def flatten_reduce(matrix):
    return list(reduce(lambda x, y: x + y, matrix, []))

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

def find_shared_nums(set1, set2, set3):
    common_numbers = set()

    common12 = set1.intersection(set2)
    common_numbers.update(common12)

    common13 = set1.intersection(set3)
    common_numbers.update(common13)

    common23 = set2.intersection(set3)
    common_numbers.update(common23)

    return common_numbers

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
    
def add_inferred_matches(guessed_hits: pd.DataFrame, id: str, retriever, remove_stopwords:bool=False) -> pd.DataFrame:
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
                    words = sermon.chunked[pars[0]][sents[0]+1]["words"]
                    if remove_stopwords:
                        filtered_words = [word for word in words if word.lower() not in em_stopwords]
                    else:
                        filtered_words = words
                    missing_sent = " ".join(filtered_words)
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
                                                        "Fundstelle", "Vers", 
                                                        "Ähnlichkeit", "Dopplung"])

        guessed_hits = pd.concat([guessed_hits, new_matches])
        guessed_hits.sort_values(["Paragraph", "Satz"], ascending=True, inplace=True)
        guessed_hits.reset_index(drop=True)
    
    return guessed_hits

def correct_inbetween_matches(df: pd.DataFrame, retriever) -> pd.DataFrame:
    for i in range(0, len(df) - 3):
        chunk = df.iloc[i:i+3]
        pages = chunk["Fundstelle"].to_list()
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

def remove_duplicates(df):
    # group hits by par-sent
    grouped_hits = df.copy().groupby(["Paragraph", "Satz"])

    group_keys = list(grouped_hits.groups.keys())

    def page_add(x): 
        if type(x) == int:
            return [x, x+1,x+2]
        else:
            splits = x.split("_")
            if len(splits) == 3:
                #book, chap, vers = x.split("_")
                book = splits[0]
                chap = splits[1]
                vers = splits[2]
                return [f"{book}_{chap}_{vers}",
                        f"{book}_{chap}_{int(vers) + 1}",
                        f"{book}_{chap}_{int(vers) + 2}"]
            else:
                print(f"Couldn't split Fundstelle: {x}")
                return [x]
    sent_add = lambda x : [x, x+1, x+2, x+3]

    for i in range(0, len(group_keys)-2):
        # iterate over group_keys in 3-grams
        keys = group_keys[i:i+3]
        all_pages = []
        if is_equal([x[0] for x in keys]):
            for key in keys:
                current_group = grouped_hits.get_group(key)
                group_pages = current_group["Fundstelle"].to_list()
                potential_pages = flatten_reduce([page_add(x) for x in group_pages])
                all_pages.append(set(potential_pages))
            shared_page = find_shared_nums(all_pages[0], all_pages[1], all_pages[2])
            if len(shared_page) > 0:
                all_right_pages = []
                all_wrong_pages = []
                for key in keys:
                    current_group = grouped_hits.get_group(key)
                    if bool(len(current_group[current_group["Fundstelle"].isin(shared_page)])):
                        right_pages = current_group[current_group["Fundstelle"].isin(shared_page)]
                        wrong_pages = current_group[~current_group["Fundstelle"].isin(shared_page)]
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
def page_add(x): 
        if type(x) == int:
            return [x, x+1,x+2]
        else:
            book, chap, vers = x.split("_")
            return [f"{book}_{chap}_{vers}",
                    f"{book}_{chap}_{int(vers) + 1}",
                    f"{book}_{chap}_{int(vers) + 2}"]
        
#%%
def find_similarities(task: str, id: str, cosine_cutoff: float, retriever, test=False, remove_stopwords:bool=False) -> pd.DataFrame:
    """_summary_

    Args:
        task (str): _description_
        id (str): _description_
        relevant_texts (list): _description_
        fuzziness (int): _description_
        test (bool, optional): _description_. Defaults to False.

    Returns:
        pd.DataFrame: _description_
    """
    if task == "bibel":
        print(f"starting with {id}")
        sermon = oa.Sermon(id)
        hits = []
        sent_nr = 0
        for i in range(len(sermon.chunked)):                # for each paragraph
                for j in range(len(sermon.chunked[i])):         # for each sentence
                    if " musikwerk" in sermon.chunked[i][j]["types"] and test:
                        continue
                    else:
                        sent_nr += 1
                        words = sermon.chunked[i][j]["words"]
                        if remove_stopwords:
                            filtered_words = [word for word in words if word.lower() not in em_stopwords]
                        else:
                            filtered_words = words
                        query = " ".join(filtered_words)
                        query = re.sub(r'[/.,;:?!]', '', query)
                        matches = retriever.invoke({"query": query})
                        for match in matches:
                            if match.metadata["score"] < cosine_cutoff:
                                hits.append([query, i, j, match.page_content, match.metadata["source"], match.metadata["score"], False])
        
        guessed_hits = pd.DataFrame(hits, columns=["Predigt", "Paragraph", "Satz", "Vers", "Fundstelle", "Ähnlichkeit", "Validated"])     # create dataframe
        guessed_hits['Dopplung'] = guessed_hits.groupby('Satz')['Satz'].transform(lambda x: x.duplicated())
        guessed_hits = remove_duplicates(guessed_hits)

        guessed_hits =guessed_hits.drop(guessed_hits[(guessed_hits['Validated'] == False) & (guessed_hits['Ähnlichkeit'] > 0.2)].index)

    else:
        print(f"starting with {id}")
        hits = []
        sermon = oa.Sermon(id)
        sent_nr = 0
        for i in range(len(sermon.chunked)):                # for each paragraph
                for j in range(len(sermon.chunked[i])):         # for each sentence
                    if " bibel" in sermon.chunked[i][j]["types"] and test:
                        continue
                    else:
                        sent_nr += 1
                        words = sermon.chunked[i][j]["words"]
                        if remove_stopwords:
                            filtered_words = [word for word in words if word.lower() not in em_stopwords]
                        else:
                            filtered_words = words
                        query = " ".join(filtered_words)
                        query = re.sub(r'[/.,;:?!]', '', query)
                        matches = retriever.invoke({"query": query})
                        for match in matches:
                            if match.metadata["score"] < cosine_cutoff:
                                hits.append([query, i, j, match.page_content, int(match.metadata["source"]), match.metadata["score"], False])
        
        guessed_hits = pd.DataFrame(hits, columns=["Predigt", "Paragraph", "Satz", "Vers", "Fundstelle", "Ähnlichkeit", "Validated"])     # create dataframe
        guessed_hits['Dopplung'] = guessed_hits.groupby('Satz')['Satz'].transform(lambda x: x.duplicated())

        guessed_hits = remove_duplicates(guessed_hits)

        guessed_hits =guessed_hits.drop(guessed_hits[(guessed_hits['Validated'] == False) & (guessed_hits['Ähnlichkeit'] > 0.2)].index)

    return guessed_hits