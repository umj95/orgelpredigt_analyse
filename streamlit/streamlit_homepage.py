import streamlit as st

import sys, os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print(">>> SYS.PATH:", sys.path)  # (optional, for debugging)


print("new path after append:")
print(sys.path)

pg = st.navigation([st.Page("pages/Startseite.py"),
                    st.Page("pages/Orgelpredigt_Analyse.py"), 
                    st.Page("pages/Orgelpredigt_Vergleich.py")])

pg.run()
