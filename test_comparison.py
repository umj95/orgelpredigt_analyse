# %%
import pandas as pd
import json
import plotly
import streamlit

# %%
with open('test_results.json', 'r') as f:
    test_results = json.load(f)

# %%
overall_result_table = [['Timestamp', 'Method', 'Fuzziness', 'Average Certainty','Precision', 'Recall', 'F1-Score']]

for result in test_results:
    item = [result['date'], result['type'], result['fuzziness'],
            result['overall_certainty'], result['overall_precision'],
            result['overall_recall'], result['overall_f1']]
    overall_result_table.append(item)

overall_results = pd.DataFrame(overall_result_table)
overall_results