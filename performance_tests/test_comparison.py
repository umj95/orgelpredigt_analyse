# %%
import pandas as pd
import json
import plotly
import streamlit

from pathlib import Path

# root directory path
ROOT = Path(__file__).resolve().parents[1]

# %%
with open(ROOT / 'test_results_longest.json', 'r') as f:
    test_results_longest = json.load(f)
with open(ROOT / 'test_results_most.json', 'r') as f:
    test_results_most = json.load(f)

# %%
overall_result_table_most = [['Timestamp', 'Method', 'Fuzziness', 'Average Certainty','Precision', 'Recall', 'F1-Score']]

for result in test_results_most:
    item = [result['date'], result['type'], result['fuzziness'],
            result['overall_certainty_verse'], result['overall_precision_verse'],
            result['overall_recall_verse'], result['overall_f1_verse']]
    overall_result_table_most.append(item)

overall_results_most = pd.DataFrame(overall_result_table_most)
overall_results_most

# %%
overall_result_table_longest = [['Timestamp', 'Method', 'Fuzziness', 'Average Certainty','Precision', 'Recall', 'F1-Score']]

for result in test_results_longest:
    item = [result['date'], result['type'], result['fuzziness'],
            result['overall_certainty_verse'], result['overall_precision_verse'],
            result['overall_recall_verse'], result['overall_f1_verse']]
    overall_result_table_longest.append(item)

overall_results_longest = pd.DataFrame(overall_result_table_longest)
overall_results_longest