import streamlit as st

import sys, os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Get the absolute path to the repository root
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the repository root to the Python path
sys.path.append(repo_root)

print("new path after append:")
print(sys.path)

pg = st.navigation([st.Page("pages/Startseite.py"),
                    st.Page("pages/Orgelpredigt_Analyse.py"), 
                    st.Page("pages/Orgelpredigt_Vergleich.py")])

pg.run()
