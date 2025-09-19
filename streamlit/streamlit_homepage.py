import streamlit as st

import sys, os

print(sys.path)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

pg = st.navigation([st.Page("pages/Startseite.py"),
                    st.Page("pages/Orgelpredigt_Analyse.py"), 
                    st.Page("pages/Orgelpredigt_Vergleich.py")])

pg.run()
