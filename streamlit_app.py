import streamlit as st
from utils import search_pubmed, retrieve_abstracts, summarize_abstracts
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
import sqlite3
import pandas as pd
import os

import subprocess
import streamlit as st

version_info = subprocess.run(["pip", "show", "streamlit-authenticator"], capture_output=True, text=True)
st.code(version_info.stdout)

# Load config
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'], config['cookie']['name'],
    config['cookie']['key'], config['cookie']['expiry_days']
)

# positional for name, keyword for location:
name, authentication_status, username = authenticator.login(name='Login', location='main')

if authentication_status:
    st.sidebar.success(f"Welcome {name}")
    authenticator.logout('Logout', 'sidebar')

    st.title("🔬 PubMed Abstract Analyzer")

    with st.expander("🧠 Search Criteria Agreement", expanded=True):
        st.write("By using this tool, you agree to use PubMed data responsibly and comply with NCBI API guidelines.")
        st.markdown("""- Use your own email address for API identification.
- Avoid excessive requests.
- Results are for research purposes only.""")

    email = st.text_input("Enter your email (used for NCBI API)", value=config['email'], key="email")

    topic = st.text_input("🔍 Search Topic", "bone remodeling")
    retmax = st.number_input("📄 Number of abstracts to retrieve", min_value=10, max_value=1000, value=100, step=10)
    do_search = st.button("Start Search")

    if do_search and email:
        st.info("Searching PubMed...")
        ids = search_pubmed(topic, email=email, retmax=retmax)
        st.success(f"Found {len(ids)} abstracts.")

        if ids:
            st.info("Retrieving abstracts with sleep intervals...")
            retrieve_abstracts(ids, email=email, db_path="abstracts.db")

            st.success("Abstracts saved to database ✅")

            if st.button("Summarize Abstracts"):
                st.info("Generating summary...")
                summary = summarize_abstracts("abstracts.db")
                st.markdown("## 📝 Summary")
                st.markdown(summary)

            if st.button("Download Abstracts"):
                conn = sqlite3.connect("abstracts.db")
                df = pd.read_sql("SELECT * FROM abstracts", conn)
                csv = df.to_csv(index=False)
                st.download_button("Download CSV", csv, file_name="abstracts.csv")
else:
    if authentication_status is False:
        st.error("Username/password is incorrect")
    elif authentication_status is None:
        st.warning("Please enter your credentials")
