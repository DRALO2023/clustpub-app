# import streamlit as st
# import sqlite3
# import bcrypt
# import pandas as pd
# import time
# from Bio import Entrez
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from transformers import pipeline
# import matplotlib.pyplot as plt
# import seaborn as sns
# import io

# plt.switch_backend("Agg")  # Fix backend for Streamlit Cloud

# # --- PubMed Search & Retrieval Functions ---
import streamlit as st
import sqlite3
import bcrypt
import pandas as pd
import time
from Bio import Entrez
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns

plt.switch_backend("Agg")  # Fix backend for Streamlit Cloud

# --- PubMed Search & Retrieval Functions ---
def search_pubmed(topic, email, retmax=100):  # ✅ Default retmax fixed
    Entrez.email = email
    handle = Entrez.esearch(db="pubmed", term=topic, retmax=1)
    record = Entrez.read(handle)
    handle.close()
    total_count = int(record["Count"])
    retmax = min(retmax, total_count)
    handle = Entrez.esearch(db="pubmed", term=topic, retmax=retmax)
    record = Entrez.read(handle)
    handle.close()
    return record['IdList'], total_count

def retrieve_abstracts(id_list, email, db_path="abstracts.db", batch_size=100):  # ✅ Default batch_size fixed
    Entrez.email = email
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("DROP TABLE IF EXISTS abstracts")
    c.execute("""CREATE TABLE IF NOT EXISTS abstracts (
                    pmid TEXT PRIMARY KEY,
                    title TEXT,
                    abstract TEXT
                )""")
    conn.commit()

    for start in range(0, len(id_list), batch_size):
        end = min(start + batch_size, len(id_list))
        ids = id_list[start:end]
        handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="xml")
        papers = Entrez.read(handle)
        handle.close()
        for paper in papers.get('PubmedArticle', []):
            try:
                pmid = paper['MedlineCitation']['PMID']
                title = paper['MedlineCitation']['Article']['ArticleTitle']
                abstract = paper['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
                c.execute("INSERT OR IGNORE INTO abstracts VALUES (?, ?, ?)", (pmid, title, abstract))
            except Exception:
                continue
        conn.commit()
        time.sleep(0.4)
    conn.close()

def summarize_abstracts(db_path="abstracts.db"):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM abstracts", conn)
    conn.close()

    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)  # ✅ Fixed max_features
    X = tfidf.fit_transform(df['abstract'])
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=5)
    df['cluster'] = kmeans.fit_predict(X)

    feature_names = tfidf.get_feature_names_out()
    cluster_titles = {}
    for clus in range(5):
        mask = (df['cluster'] == clus).values
        cluster_docs = X[mask]
        mean = cluster_docs.mean(axis=0).A1
        top_words = [feature_names[i] for i in mean.argsort()[::-1][:3]]
        cluster_titles[clus] = ", ".join(top_words)

    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    summaries = {}
    for clus in sorted(df['cluster'].unique()):
        clus_df = df[df['cluster'] == clus]
        text = " ".join(clus_df['abstract'].tolist())[:3000]
        summary = summarizer(text, max_length=512, min_length=200, do_sample=False)[0]['summary_text']
        summaries[clus] = summary

    # PCA plot
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X.toarray())
    df_plot = pd.DataFrame(reduced, columns=["x", "y"])
    df_plot['cluster'] = df['cluster'].map(cluster_titles)

    plt.figure()
    sns.scatterplot(data=df_plot, x='x', y='y', hue='cluster', palette="tab10")  # ✅ Fixed palette
    plt.title("PCA Cluster Plot")
    st.pyplot(plt)
    plt.close()

    # t-SNE plot
    tsne = TSNE(n_components=2, perplexity=min(30, len(df)-1), random_state=42)
    reduced_tsne = tsne.fit_transform(X.toarray())
    df_plot_tsne = pd.DataFrame(reduced_tsne, columns=["x", "y"])
    df_plot_tsne['cluster'] = df['cluster'].map(cluster_titles)

    plt.figure()
    sns.scatterplot(data=df_plot_tsne, x='x', y='y', hue='cluster', palette="tab10")  # ✅ Fixed palette
    plt.title("t-SNE Cluster Plot")
    st.pyplot(plt)
    plt.close()

    return df, cluster_titles, summaries

# --- Authentication ---
def create_users_table():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
                 username TEXT PRIMARY KEY,
                 password_hash BLOB,
                 email TEXT)""")
    conn.commit()
    conn.close()

def add_user(username, password, email):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    try:
        c.execute("INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)",
                  (username, hashed, email))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return False
    conn.close()
    return True

def authenticate_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result:
        return bcrypt.checkpw(password.encode(), result[0])
    return False

create_users_table()

# --- Streamlit App ---
st.set_page_config(page_title="PubMed Analyzer", layout="wide")

for key, default in {
    'logged_in': False, 'username': '', 'ids': None, 'total_count': None,
    'retmax': 100, 'df': None, 'summaries': None, 'titles': None, 'retrieved': False
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.title("🔐 PubMed Abstract Analyzer with Authentication")
menu = ["Login", "Sign up"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Sign up":
    st.subheader("Create New Account")
    new_user = st.text_input("Username")
    new_email = st.text_input("Email")
    new_password = st.text_input("Password", type='password')
    confirm = st.text_input("Confirm Password", type='password')
    if st.button("Sign Up"):
        if new_password != confirm:
            st.error("Passwords do not match")
        elif add_user(new_user, new_password, new_email):
            st.success("Account created successfully. Please login.")
        else:
            st.error("Username already exists.")

elif choice == "Login":
    st.subheader("Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type='password')
    if st.button("Login"):
        if authenticate_user(user, pwd):
            st.session_state.logged_in = True
            st.session_state.username = user
            st.success(f"Welcome {user}!")
        else:
            st.error("Invalid credentials")

if st.session_state.logged_in:
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    if st.sidebar.button("New Search"):
        for key in list(st.session_state.keys()):
            if key not in ["logged_in", "username"]:
                del st.session_state[key]
        st.rerun()

    st.sidebar.success(f"Logged in as {st.session_state.username}")

    with st.expander("🧠 Search Criteria Agreement", expanded=True):
        st.write("By using this tool, you agree to use PubMed data responsibly.")
        st.markdown("- Use your email address.  \n- Do not abuse the API.  \n- Research purpose only.")

    email = st.text_input("Enter your email for PubMed API", "")
    topic = st.text_input("🔍 Search Topic", "bone remodeling")

    if not st.session_state.ids:
        if st.button("Start Search") and email and topic:
            st.info("Searching PubMed...")
            ids, total_count = search_pubmed(topic, email=email, retmax=100)  # ✅ Fixed retmax
            st.session_state.ids = ids
            st.session_state.total_count = total_count
            st.success(f"Found {total_count} abstracts.")

    if st.session_state.ids:
        retmax = st.slider(
            "Number of abstracts to retrieve",
            min_value=10,
            max_value=min(st.session_state.total_count, 1000),
            value=st.session_state.retmax,
            step=10
        )
        st.session_state.retmax = retmax

        if st.button("Retrieve Abstracts"):
            with st.spinner("Retrieving abstracts... This may take a few minutes"):
                retrieve_abstracts(st.session_state.ids[:retmax], email)
            st.session_state.retrieved = True
            st.success("Abstracts retrieved and saved in the database.")

        if st.session_state.retrieved:
            if st.button("Summarize & Cluster Abstracts"):
                df, cluster_titles, summaries = summarize_abstracts()
                st.session_state.df = df
                st.session_state.titles = cluster_titles
                st.session_state.summaries = summaries

            if st.session_state.df is not None:
                st.subheader("Cluster Summaries:")
                for clus, title in st.session_state.titles.items():
                    st.markdown(f"**Cluster {clus}: {title}**")
                    st.write(st.session_state.summaries[clus])

                st.subheader("Abstracts Data (first 10 rows):")  # ✅ Fixed title
                st.dataframe(st.session_state.df.head(10))       # ✅ Limited rows

                csv = st.session_state.df.to_csv(index=False)
                st.download_button("Download abstracts as CSV", csv, file_name="abstracts.csv", mime="text/csv")

# def search_pubmed(topic, email, retmax=0):
#     Entrez.email = email
#     handle = Entrez.esearch(db="pubmed", term=topic, retmax=1)
#     record = Entrez.read(handle)
#     handle.close()
#     total_count = int(record["Count"])
#     retmax = min(retmax, total_count)
#     handle = Entrez.esearch(db="pubmed", term=topic, retmax=retmax)
#     record = Entrez.read(handle)
#     handle.close()
#     return record['IdList'], total_count

# def retrieve_abstracts(id_list, email, db_path="abstracts.db", batch_size=0):
#     Entrez.email = email
#     conn = sqlite3.connect(db_path)
#     c = conn.cursor()

#     c.execute("DROP TABLE IF EXISTS abstracts")
#     c.execute("""CREATE TABLE IF NOT EXISTS abstracts (
#                     pmid TEXT PRIMARY KEY,
#                     title TEXT,
#                     abstract TEXT
#                 )""")
#     conn.commit()

#     for start in range(0, len(id_list), batch_size):
#         end = min(start + batch_size, len(id_list))
#         ids = id_list[start:end]
#         handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="xml")
#         papers = Entrez.read(handle)
#         handle.close()
#         for paper in papers.get('PubmedArticle', []):
#             try:
#                 pmid = paper['MedlineCitation']['PMID']
#                 title = paper['MedlineCitation']['Article']['ArticleTitle']
#                 abstract = paper['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
#                 c.execute("INSERT OR IGNORE INTO abstracts VALUES (?, ?, ?)", (pmid, title, abstract))
#             except Exception:
#                 continue
#         conn.commit()
#         time.sleep(0.4)
#     conn.close()

# def summarize_abstracts(db_path="abstracts.db"):
#     conn = sqlite3.connect(db_path)
#     df = pd.read_sql("SELECT * FROM abstracts", conn)
#     conn.close()

#     tfidf = TfidfVectorizer(stop_words='english', max_features=00)
#     X = tfidf.fit_transform(df['abstract'])
#     kmeans = KMeans(n_clusters=5, random_state=42, n_init=5)
#     df['cluster'] = kmeans.fit_predict(X)

#     feature_names = tfidf.get_feature_names_out()
#     cluster_titles = {}
#     for clus in range(5):
#         mask = (df['cluster'] == clus).values
#         cluster_docs = X[mask]
#         mean = cluster_docs.mean(axis=0).A1
#         top_words = [feature_names[i] for i in mean.argsort()[::-1][:3]]
#         cluster_titles[clus] = ", ".join(top_words)

#     summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
#     summaries = {}
#     for clus in sorted(df['cluster'].unique()):
#         clus_df = df[df['cluster'] == clus]
#         text = " ".join(clus_df['abstract'].tolist())[:3000]
#         summary = summarizer(text, max_length=512, min_length=200, do_sample=False)[0]['summary_text']
#         summaries[clus] = summary

#     # PCA plot
#     pca = PCA(n_components=2)
#     reduced = pca.fit_transform(X.toarray())
#     df_plot = pd.DataFrame(reduced, columns=["x", "y"])
#     df_plot['cluster'] = df['cluster'].map(cluster_titles)

#     plt.figure()
#     sns.scatterplot(data=df_plot, x='x', y='y', hue='cluster', palette="tab")
#     plt.title("PCA Cluster Plot")
#     st.pyplot(plt)
#     plt.close()

#     # t-SNE plot
#     tsne = TSNE(n_components=2, perplexity=min(30, len(df)-1), random_state=42)
#     reduced_tsne = tsne.fit_transform(X.toarray())
#     df_plot_tsne = pd.DataFrame(reduced_tsne, columns=["x", "y"])
#     df_plot_tsne['cluster'] = df['cluster'].map(cluster_titles)

#     plt.figure()
#     sns.scatterplot(data=df_plot_tsne, x='x', y='y', hue='cluster', palette="tab")
#     plt.title("t-SNE Cluster Plot")
#     st.pyplot(plt)
#     plt.close()

#     return df, cluster_titles, summaries

# # --- Authentication ---
# def create_users_table():
#     conn = sqlite3.connect('users.db')
#     c = conn.cursor()
#     c.execute("""CREATE TABLE IF NOT EXISTS users (
#                  username TEXT PRIMARY KEY,
#                  password_hash BLOB,
#                  email TEXT)""")
#     conn.commit()
#     conn.close()

# def add_user(username, password, email):
#     conn = sqlite3.connect('users.db')
#     c = conn.cursor()
#     hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
#     try:
#         c.execute("INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)",
#                   (username, hashed, email))
#         conn.commit()
#     except sqlite3.IntegrityError:
#         conn.close()
#         return False
#     conn.close()
#     return True

# def authenticate_user(username, password):
#     conn = sqlite3.connect('users.db')
#     c = conn.cursor()
#     c.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
#     result = c.fetchone()
#     conn.close()
#     if result:
#         return bcrypt.checkpw(password.encode(), result[0])
#     return False

# create_users_table()

# # --- Streamlit App ---
# st.set_page_config(page_title="PubMed Analyzer", layout="wide")

# for key, default in {
#     'logged_in': False, 'username': '', 'ids': None, 'total_count': None,
#     'retmax': 0, 'df': None, 'summaries': None, 'titles': None, 'retrieved': False
# }.items():
#     if key not in st.session_state:
#         st.session_state[key] = default

# st.title("🔐 PubMed Abstract Analyzer with Authentication")
# menu = ["Login", "Sign up"]
# choice = st.sidebar.selectbox("Menu", menu)

# if choice == "Sign up":
#     st.subheader("Create New Account")
#     new_user = st.text_input("Username")
#     new_email = st.text_input("Email")
#     new_password = st.text_input("Password", type='password')
#     confirm = st.text_input("Confirm Password", type='password')
#     if st.button("Sign Up"):
#         if new_password != confirm:
#             st.error("Passwords do not match")
#         elif add_user(new_user, new_password, new_email):
#             st.success("Account created successfully. Please login.")
#         else:
#             st.error("Username already exists.")

# elif choice == "Login":
#     st.subheader("Login")
#     user = st.text_input("Username")
#     pwd = st.text_input("Password", type='password')
#     if st.button("Login"):
#         if authenticate_user(user, pwd):
#             st.session_state.logged_in = True
#             st.session_state.username = user
#             st.success(f"Welcome {user}!")
#         else:
#             st.error("Invalid credentials")

# if st.session_state.logged_in:
#     if st.sidebar.button("Logout"):
#         for key in list(st.session_state.keys()):
#             del st.session_state[key]
#         st.rerun()

#     if st.sidebar.button("New Search"):
#         for key in list(st.session_state.keys()):
#             if key not in ["logged_in", "username"]:
#                 del st.session_state[key]
#         st.rerun()

#     st.sidebar.success(f"Logged in as {st.session_state.username}")

#     with st.expander("🧠 Search Criteria Agreement", expanded=True):
#         st.write("By using this tool, you agree to use PubMed data responsibly.")
#         st.markdown("- Use your email address.  \n- Do not abuse the API.  \n- Research purpose only.")

#     email = st.text_input("Enter your email for PubMed API", "")
#     topic = st.text_input("🔍 Search Topic", "bone remodeling")

#     if not st.session_state.ids:
#         if st.button("Start Search") and email and topic:
#             st.info("Searching PubMed...")
#             ids, total_count = search_pubmed(topic, email=email, retmax=000)
#             st.session_state.ids = ids
#             st.session_state.total_count = total_count
#             st.success(f"Found {total_count} abstracts.")

#     if st.session_state.ids:
#         retmax = st.slider("Number of abstracts to retrieve", min_value=,
#                            max_value=st.session_state.total_count, value=st.session_state.retmax, step=)
#         st.session_state.retmax = retmax

#         if st.button("Retrieve Abstracts"):
#             with st.spinner("Retrieving abstracts... This may take a few minutes"):
#                 retrieve_abstracts(st.session_state.ids[:retmax], email)
#             st.session_state.retrieved = True
#             st.success("Abstracts retrieved and saved in the database.")

#         if st.session_state.retrieved:
#             if st.button("Summarize & Cluster Abstracts"):
#                 df, cluster_titles, summaries = summarize_abstracts()
#                 st.session_state.df = df
#                 st.session_state.titles = cluster_titles
#                 st.session_state.summaries = summaries

#             if st.session_state.df is not None:
#                 st.subheader("Cluster Summaries:")
#                 for clus, title in st.session_state.titles.items():
#                     st.markdown(f"**Cluster {clus}: {title}**")
#                     st.write(st.session_state.summaries[clus])

#                 st.subheader("Abstracts Data (first  rows):")
#                 st.dataframe(st.session_state.df.head())

#                 # Allow downloading abstracts as CSV
#                 csv = st.session_state.df.to_csv(index=False)
#                 st.download_button("Download abstracts as CSV", csv, file_name="abstracts.csv", mime="text/csv")

