
import re
import core.utils as oa
from rapidfuzz import fuzz
import pandas as pd
import os

from numpyencoder import NumpyEncoder
from pathlib import Path

# root directory path
ROOT = Path(os.getcwd()).resolve().parents[0]

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
    if 'Bibelstelle' in df:
        for satz_id, indices in find_duplicate_satz(df.copy()).items():
            satz = [int(x) for x in satz_id.split("-")]
            if indices[0]-1 in df.index:
                check = df["Bibelstelle"][indices[0]-1]
            else:
                check = df["Bibelstelle"][indices[-1]+1]
            print(check)
            matches = df.query(f"Paragraph == {satz[0]} and Satz == {satz[1]} and Bibelstelle == '{check}'").index
            if len(matches):
                match_index = matches[0]
                for i in indices:
                    if i != match_index:
                        df.drop([i], inplace=True)
    else:
        for satz_id, indices in find_duplicate_satz(df.copy()).items():
            satz = [int(x) for x in satz_id.split("-")]
            if indices[0]-1 in df.index:
                check = df["Liederbuch"][indices[0]-1]
            else:
                check = df["Liederbuch"][indices[-1]+1]
            print(check)
            matches = df.query(f"Paragraph == {satz[0]} and Satz == {satz[1]} and Liederbuch == '{check}'").index
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


def find_similarities(task: str, sermons: list, relevant_texts: list, fuzziness: int) -> list:
    """Find passages from a list of texts in a list of sermons  

    Args:
        task (str): "lieder", or "bibel"
        sermons (list): list of sermons in which similarities are to be searched
        relevant_texts (list): list of texts in which similarities are to be found
        fuzziness (int): The fuzziness factor

    Returns:
        list: a list of hits in csv form for each sermon
    """
    results = []
    if task == "lieder":
        for id in sermons:
            print(f"Starting with {id}")
            sermon = oa.Sermon(id)

            # perform classification
            hits = []
            all_sents = 0
            for i in range(len(sermon.chunked)):                # for each paragraph
                for j in range(len(sermon.chunked[i])):         # for each sentence
                    if " bibel" in sermon.chunked[i][j]["types"]:
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

            guessed_hits = pd.DataFrame(hits, columns=["Predigt", "Paragraph", "Satz", "Liederbuch", "Liedvers", "Ähnlichkeit"])     # create dataframe
            guessed_hits['Dopplung'] = guessed_hits.groupby('Satz')['Satz'].transform(lambda x: x.duplicated())

            guessed_hits = remove_duplicates(guessed_hits).reset_index(drop=True)

            results.append([id, all_sents, guessed_hits.to_csv()[1:]])
    else:
        for id in sermons:
            print(f"Starting with {id}")
            sermon = oa.Sermon(id)

            # perform classification
            hits = []
            all_sents = 0
            for i in range(len(sermon.chunked)):                # for each paragraph
                for j in range(len(sermon.chunked[i])):         # for each sentence
                    if " musikwerk" in sermon.chunked[i][j]["types"]:
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

            guessed_hits = pd.DataFrame(hits, columns=["Predigt", "Paragraph", "Satz", "Bibelstelle", "Bibelvers", "Ähnlichkeit"])     # create dataframe
            guessed_hits['Dopplung'] = guessed_hits.groupby('Satz')['Satz'].transform(lambda x: x.duplicated())

            guessed_hits = remove_duplicates(guessed_hits).reset_index(drop=True)

            results.append([id, all_sents, guessed_hits.to_csv()[1:]])
    
    return results
