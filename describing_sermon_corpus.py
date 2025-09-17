# %%
import pandas as pd
import os
import orgelpredigt_analysis as oa
import json
from collections import Counter
import re
import plotly.graph_objects as go
from pprint import pprint

# %%
with open('predigten_übersicht.json') as f:
    predigten = json.load(f)

# %%
musik_id = re.compile(r'E10[0-9]{4}')
    
# %%
# expand the info in predigten_übersicht.json
all_words = []
all_types = []
all_refs = []

quantities = []

for id in [*predigten]:
    sermon = oa.Sermon(id)

    all_words += sermon.words
    all_types += sermon.word_types
    all_refs += sermon.all_references

    predigten[id]["length"] = len(sermon.words)
    predigten[id]["worte_bibel"] = Counter(sermon.word_types)[' bibel']
    predigten[id]["worte_quellen"] = Counter(sermon.word_types)[' quelle'] + Counter(sermon.word_types)[' literatur']
    predigten[id]["worte_orgelpredigt"] = Counter(sermon.word_types)[' orgelpredigt']
    predigten[id]["worte_musikwerk"] = Counter(sermon.word_types)[' musikwerk']
    predigten[id]["zitierte_musikwerke"] = len(list(set([s for s in sermon.all_references if musik_id.match(s)])))

with open("predigten_übersicht.json", "w") as f:
    json.dump(predigten, f, ensure_ascii=False)

# %%
# print info about aggregated word, type and ref counts.
print(f"Nr. der Worte in allen Predigten: {len(all_words)}")
type_counters = dict(Counter(all_types))
print(f"Nr. der Wörter in den Zitationstypen:")
pprint(type_counters)

# %%
len(predigten)

# %%
# plot the 20 reference works with the highest word count overall
ref_counters = dict(Counter(all_refs).most_common(20))
refs_only_ids = {oa.get_short_info(key): value for key, value in ref_counters.items() if re.match(r'E[01][0-9]{5}', str(key))}

fig = go.Figure(data=[go.Bar(x=list(refs_only_ids.keys()), 
                             y=list(refs_only_ids.values()), 
                             marker_color='blue', 
                             marker_line_color='blue')])

fig.update_layout(title='20 meistzitierte Quellen, Musikwerke & Predigten in Orgelpredigten', 
                  xaxis_title='Werke', 
                  yaxis_title='Worte')
fig.show()

# %%
# plot the quoted Songs
ref_counters = dict(Counter(all_refs))

ref_counter_music = {oa.get_short_info(key): value for key, value in ref_counters.items() if key.startswith("E10")}

#ref_counter_music.pop("no_composer: no_title")
ref_counter_music.pop("E100: E100")

fig = go.Figure(data=[go.Bar(x=list(ref_counter_music.keys()), 
                             y=list(ref_counter_music.values()), 
                             marker_color='blue', 
                             marker_line_color='blue')])

fig.update_layout(title='Zitierte Musikwerke', 
                  xaxis_title='Werke', 
                  yaxis_title='Worte')
fig.show()

# %%
print(refs_only_ids)
# %%
# create table of most quoted works
df = pd.DataFrame.from_dict({k:[v] for k,v in refs_only_ids.items()})
df

# %%
# plot the 20 most quoted songs
import heapq


top_20 = heapq.nlargest(20, ref_counter_music.items(), key=lambda x: x[1])
top_20_music = {i[0]: i[1] for i in top_20}

fig = go.Figure(data=[go.Bar(x=list(top_20_music.keys()), 
                             y=list(top_20_music.values()), 
                             marker_color='blue', 
                             marker_line_color='blue')])

fig.update_layout(title='20 meistzitierte Musikwerke', 
                  xaxis_title='Werke', 
                  yaxis_title='Worte')
fig.show()

# %%
print(top_20_music)

# %%
# plot the 5 most quoted sermons
ref_counter_sermons = {oa.get_short_info(key).split(":")[1]: value for key, value in ref_counters.items() if key.startswith("E00")}

import heapq


top_5 = heapq.nlargest(5, ref_counter_sermons.items(), key=lambda x: x[1])
top_5_sermons = {i[0]: i[1] for i in top_5}

fig = go.Figure(data=[go.Bar(x=list(top_5_sermons.keys()), 
                             y=list(top_5_sermons.values()), 
                             marker_color='blue', 
                             marker_line_color='blue')])

fig.update_layout(title='5 meistzitierte Orgelpredigten', 
                  xaxis_title='Predigten', 
                  yaxis_title='Worte in anderen Predigten')
fig.show()


# %%
# Find 5 sermons with longest music quotes
top_5_most_music_words = heapq.nlargest(5, predigten, key=lambda x: predigten[x]['worte_musikwerk'])

print("=== Die top 5 Predigten mit den längsten Musikzitaten ===")

print(*[f"{oa.get_short_info(i)} [{i}]" for i in top_5_most_music_words], sep="\n")

# %%
# Find 5 sermons with most pieces of music quoted
top_5_most_music_pieces = heapq.nlargest(5, predigten, key=lambda x: predigten[x]['zitierte_musikwerke'])

print("=== Die top 5 Predigten mit den meisten Musikzitaten ===")

print(*[f"{oa.get_short_info(i)} [{i}]" for i in top_5_most_music_pieces], sep="\n")

# %%
# Find 5 sermons with longest quotes from other sermons
top_5_most_sermon_quotes = heapq.nlargest(5, predigten, key=lambda x: predigten[x]['worte_orgelpredigt'])

print("=== Die top 5 Predigten mit den längsten Zitaten aus anderen Orgelpredigten ===")

print(*[f"{oa.get_short_info(i)} [{i}]" for i in top_5_most_sermon_quotes], sep="\n")

# %%
orgelpredigt_zitate = []
for id, info in predigten.items():
    sermon = oa.Sermon(id)
    predigtzitate = len(sermon.orgelpredigtzitate)
    orgelpredigt_zitate.append([sermon.kurztitel, sermon.predigtzitate])

#%%
orgelpredigt_zitate_sorted =sorted(orgelpredigt_zitate, key=lambda x: x[1], reverse=True)
orgelpredigt_zitate_sorted

# %%
total_songquotes = 0
for id, predigt in predigten.items():
    total_songquotes += predigt["zitierte_musikwerke"]

print(f"{total_songquotes} in 65 Predigten, also im Durchschnitt {total_songquotes / 65}")