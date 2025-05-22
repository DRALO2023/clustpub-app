import streamlit as st
import sqlite3
import bcrypt
import pandas as pd
from utils import search_pubmed, retrieve_abstracts, summarize_abstracts
import base64
import os

# --- Auth DB ---
def create_users_table():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash BLOB,
            email TEXT
        )
    """)
    conn.commit()
    conn.close()

def add_user(username, password, email):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    try:
        c.execute("INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)", (username, hashed, email))
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

# --- Session State Init ---
st.set_page_config(page_title="PubMed Analyzer", layout="wide")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ''
if 'ids' not in st.session_state:
    st.session_state.ids = None
if 'total_count' not in st.session_state:
    st.session_state.total_count = None
if 'retmax' not in st.session_state:
    st.session_state.retmax = 100
if 'df' not in st.session_state:
    st.session_state.df = None
if 'summaries' not in st.session_state:
    st.session_state.summaries = None
if 'titles' not in st.session_state:
    st.session_state.titles = None
if 'retrieved' not in st.session_state:
    st.session_state.retrieved = False

# --- UI ---
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

# --- Main App ---
if st.session_state.logged_in:

    if st.sidebar.button("Logout"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.experimental_rerun()

    st.sidebar.success(f"Logged in as {st.session_state.username}")

    with st.expander("🧠 Search Criteria Agreement", expanded=True):
        st.write("By using this tool, you agree to use PubMed data responsibly.")
        st.markdown("- Use your email address.\n- Do not abuse the API.\n- Research purpose only.")

    email = st.text_input("Enter your email for PubMed API", "")
    topic = st.text_input("🔍 Search Topic", "bone remodeling")

    if not st.session_state.ids:
        if st.button("Start Search") and email and topic:
            st.info("Searching PubMed...")
            ids, total_count = search_pubmed(topic, email=email, retmax=10000)
            st.session_state.ids = ids
            st.session_state.total_count = total_count
            st.success(f"Found {total_count} abstracts.")

    if st.session_state.ids:
        retmax = st.slider("Number of abstracts to retrieve", min_value=10,
                           max_value=st.session_state.total_count, value=st.session_state.retmax, step=10)
        st.session_state.retmax = retmax

        if st.button("Retrieve Abstracts"):
            st.info(f"Retrieving {retmax} abstracts...")
            retrieve_abstracts(st.session_state.ids[:retmax], email=email, db_path="abstracts.db")
            st.success("Abstracts saved to database ✅")
            st.session_state.retrieved = True

    if st.session_state.retrieved:
        if st.button("📊 Analyze & Visualize"):
            df, titles, summaries = summarize_abstracts("abstracts.db")
            st.session_state.df = df
            st.session_state.summaries = summaries
            st.session_state.titles = titles

        if st.session_state.df is not None:
            st.markdown("### 🧠 Clustered Abstracts")
            st.dataframe(st.session_state.df[['pmid', 'title', 'cluster']])

            if os.path.exists("pca_plot.png"):
                st.image("pca_plot.png", caption="PCA Plot")
            if os.path.exists("tsne_plot.png"):
                st.image("tsne_plot.png", caption="t-SNE Plot")

            st.markdown("### 📝 Summarized Clusters")
            markdown = ""
            for i in sorted(st.session_state.summaries.keys()):
                title = st.session_state.titles[i]
                body = st.session_state.summaries[i]
                st.markdown(f"#### 🧩 {title}")
                st.write(body)
                markdown += f"### {title}\n\n{body}\n\n"

            # Markdown preview
            with st.expander("📘 Preview Markdown Summary"):
                st.markdown(markdown)

            # Markdown download
            b64 = base64.b64encode(markdown.encode()).decode()
            href = f'<a href="data:file/markdown;base64,{b64}" download="pubmed_summary.md">📄 Download Markdown Summary</a>'
            st.markdown(href, unsafe_allow_html=True)

            # ZIP file download (if exists)
            if os.path.exists("pubmed_review_results.zip"):
                with open("pubmed_review_results.zip", "rb") as f:
                    st.download_button("📦 Download ZIP File", f, file_name="pubmed_review_results.zip")

    if st.button("📥 Download Raw Abstracts"):
        conn = sqlite3.connect("abstracts.db")
        df = pd.read_sql("SELECT * FROM abstracts", conn)
        conn.close()
        csv = df.to_csv(index=False)
        st.download_button("Download CSV", csv, file_name="abstracts.csv")

# import streamlit as st
# import sqlite3
# import bcrypt
# from utils import search_pubmed, retrieve_abstracts, summarize_abstracts
# import pandas as pd

# # --- Auth DB ---
# def create_users_table():
#     conn = sqlite3.connect('users.db')
#     c = conn.cursor()
#     c.execute("""
#         CREATE TABLE IF NOT EXISTS users (
#             username TEXT PRIMARY KEY,
#             password_hash BLOB,
#             email TEXT
#         )
#     """)
#     conn.commit()
#     conn.close()

# def add_user(username, password, email):
#     conn = sqlite3.connect('users.db')
#     c = conn.cursor()
#     hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
#     try:
#         c.execute("INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)", (username, hashed, email))
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

# # --- Streamlit UI ---
# st.title("🔐 PubMed Abstract Analyzer with Auth")

# if 'logged_in' not in st.session_state:
#     st.session_state.logged_in = False
# if 'username' not in st.session_state:
#     st.session_state.username = ''
# if 'ids' not in st.session_state:
#     st.session_state.ids = None
# if 'total_count' not in st.session_state:
#     st.session_state.total_count = None
# if 'retmax' not in st.session_state:
#     st.session_state.retmax = 100
# if 'df' not in st.session_state:
#     st.session_state.df = None
# if 'summaries' not in st.session_state:
#     st.session_state.summaries = None
# if 'retrieved' not in st.session_state:
#     st.session_state.retrieved = False

# menu = ["Login", "Sign up"]
# choice = st.sidebar.selectbox("Menu", menu)

# if choice == "Sign up":
#     st.subheader("Create New Account")
#     new_user = st.text_input("Username")
#     new_email = st.text_input("Email")
#     new_password = st.text_input("Password", type='password')
#     new_password_confirm = st.text_input("Confirm Password", type='password')
#     if st.button("Sign Up"):
#         if new_password != new_password_confirm:
#             st.error("Passwords do not match")
#         elif add_user(new_user, new_password, new_email):
#             st.success("Account created successfully. Please login.")
#         else:
#             st.error("Username already exists.")

# elif choice == "Login":
#     st.subheader("Login to your account")
#     username = st.text_input("Username")
#     password = st.text_input("Password", type='password')
#     if st.button("Login"):
#         if authenticate_user(username, password):
#             st.session_state.logged_in = True
#             st.session_state.username = username
#             st.success(f"Welcome {username}!")
#         else:
#             st.error("Invalid username or password")

# # --- Main App ---
# if st.session_state.logged_in:
#     if st.sidebar.button("Logout"):
#         st.session_state.logged_in = False
#         st.session_state.username = ''
#         st.experimental_rerun()

#     st.sidebar.success(f"Logged in as {st.session_state.username}")

#     with st.expander("🧠 Search Criteria Agreement", expanded=True):
#         st.write("By using this tool, you agree to use PubMed data responsibly and comply with NCBI API guidelines.")
#         st.markdown("""- Use your own email address for API identification.
# - Avoid excessive requests.
# - Results are for research purposes only.""")

#     email = st.text_input("Enter your email (used for NCBI API)", value="")
#     topic = st.text_input("🔍 Search Topic", "bone remodeling")

#     if not st.session_state.ids:
#         do_search = st.button("Start Search")
#     else:
#         do_search = False

#     if do_search and email and topic:
#         st.info("Searching PubMed...")
#         ids, total_count = search_pubmed(topic, email=email, retmax=10000)
#         st.session_state.ids = ids
#         st.session_state.total_count = total_count
#         st.session_state.retrieved = False
#         st.success(f"Found {total_count} abstracts.")

#     if st.session_state.ids:
#         retmax = st.slider("Number of abstracts to retrieve", min_value=10, max_value=st.session_state.total_count, value=st.session_state.retmax, step=10)
#         st.session_state.retmax = retmax

#         if st.button("Retrieve Abstracts"):
#             st.info(f"Retrieving {retmax} abstracts from PubMed...")
#             retrieve_abstracts(st.session_state.ids[:retmax], email=email, db_path="abstracts.db")
#             st.success("Abstracts saved to database ✅")
#             st.session_state.retrieved = True

#     if st.session_state.retrieved:
#         if st.button("📊 Analyze & Visualize"):
#             df, cluster_titles, summaries = summarize_abstracts("abstracts.db")
#             st.session_state.df = df
#             st.session_state.summaries = summaries

#         if st.session_state.df is not None:
#             st.markdown("### 🧠 Clustered Abstracts")
#             st.dataframe(st.session_state.df[['pmid', 'title', 'cluster']])

#             st.image("pca_plot.png", caption="PCA Cluster Plot")
#             st.image("tsne_plot.png", caption="t-SNE Cluster Plot")

#             st.markdown("### 📝 Summary")
#             for summary in st.session_state.summaries:
#                 st.markdown(summary)

#             with open("pubmed_review_results.zip", "rb") as f:
#                 st.download_button("📥 Download ZIP", f, file_name="pubmed_review_results.zip")

#     if st.button("📥 Download Raw Abstracts"):
#         conn = sqlite3.connect("abstracts.db")
#         df = pd.read_sql("SELECT * FROM abstracts", conn)
#         conn.close()
#         csv = df.to_csv(index=False)
#         st.download_button("Download CSV", csv, file_name="abstracts.csv")

# # import streamlit as st
# # import sqlite3
# # import bcrypt
# # from utils import search_pubmed, retrieve_abstracts, summarize_abstracts
# # import pandas as pd

# # # --- Auth DB ---
# # def create_users_table():
# #     conn = sqlite3.connect('users.db')
# #     c = conn.cursor()
# #     c.execute("""
# #         CREATE TABLE IF NOT EXISTS users (
# #             username TEXT PRIMARY KEY,
# #             password_hash BLOB,
# #             email TEXT
# #         )
# #     """)
# #     conn.commit()
# #     conn.close()

# # def add_user(username, password, email):
# #     conn = sqlite3.connect('users.db')
# #     c = conn.cursor()
# #     hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
# #     try:
# #         c.execute("INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)", (username, hashed, email))
# #         conn.commit()
# #     except sqlite3.IntegrityError:
# #         conn.close()
# #         return False
# #     conn.close()
# #     return True

# # def authenticate_user(username, password):
# #     conn = sqlite3.connect('users.db')
# #     c = conn.cursor()
# #     c.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
# #     result = c.fetchone()
# #     conn.close()
# #     if result:
# #         return bcrypt.checkpw(password.encode(), result[0])
# #     return False

# # create_users_table()

# # # --- Streamlit UI ---
# # st.title("🔐 PubMed Abstract Analyzer with Auth")

# # if 'logged_in' not in st.session_state:
# #     st.session_state.logged_in = False
# # if 'username' not in st.session_state:
# #     st.session_state.username = ''
# # if 'ids' not in st.session_state:
# #     st.session_state.ids = None
# # if 'total_count' not in st.session_state:
# #     st.session_state.total_count = None
# # if 'retmax' not in st.session_state:
# #     st.session_state.retmax = 100
# # if 'df' not in st.session_state:
# #     st.session_state.df = None
# # if 'summaries' not in st.session_state:
# #     st.session_state.summaries = None
# # if 'retrieved' not in st.session_state:
# #     st.session_state.retrieved = False

# # menu = ["Login", "Sign up"]
# # choice = st.sidebar.selectbox("Menu", menu)

# # if choice == "Sign up":
# #     st.subheader("Create New Account")
# #     new_user = st.text_input("Username")
# #     new_email = st.text_input("Email")
# #     new_password = st.text_input("Password", type='password')
# #     new_password_confirm = st.text_input("Confirm Password", type='password')
# #     if st.button("Sign Up"):
# #         if new_password != new_password_confirm:
# #             st.error("Passwords do not match")
# #         elif add_user(new_user, new_password, new_email):
# #             st.success("Account created successfully. Please login.")
# #         else:
# #             st.error("Username already exists.")

# # elif choice == "Login":
# #     st.subheader("Login to your account")
# #     username = st.text_input("Username")
# #     password = st.text_input("Password", type='password')
# #     if st.button("Login"):
# #         if authenticate_user(username, password):
# #             st.session_state.logged_in = True
# #             st.session_state.username = username
# #             st.success(f"Welcome {username}!")
# #         else:
# #             st.error("Invalid username or password")

# # # --- Main App ---
# # if st.session_state.logged_in:
# #     if st.sidebar.button("Logout"):
# #         st.session_state.logged_in = False
# #         st.session_state.username = ''
# #         st.experimental_rerun()

# #     st.sidebar.success(f"Logged in as {st.session_state.username}")

# #     with st.expander("🧠 Search Criteria Agreement", expanded=True):
# #         st.write("By using this tool, you agree to use PubMed data responsibly and comply with NCBI API guidelines.")
# #         st.markdown("""- Use your own email address for API identification.
# # - Avoid excessive requests.
# # - Results are for research purposes only.""")

# #     email = st.text_input("Enter your email (used for NCBI API)", value="")
# #     topic = st.text_input("🔍 Search Topic", "bone remodeling")
    
# #     if not st.session_state.ids:  # Avoid rerun if IDs are already retrieved
# #         do_search = st.button("Start Search")
# #     else:
# #         do_search = False

# #     # If search button clicked and email/topic provided, retrieve abstracts
# #     if do_search and email and topic:
# #         st.info("Searching PubMed...")
# #         ids, total_count = search_pubmed(topic, email=email, retmax=10000)
# #         st.session_state.ids = ids
# #         st.session_state.total_count = total_count
# #         st.session_state.retrieved = False  # Reset retrieved flag
# #         st.success(f"Found {total_count} abstracts.")

# #     # Only show slider if abstracts are found
# #     if st.session_state.ids:
# #         retmax = st.slider("Number of abstracts to retrieve", min_value=10, max_value=st.session_state.total_count, value=st.session_state.retmax, step=10)
# #         st.session_state.retmax = retmax  # Save retmax

# #         if st.button("Retrieve Abstracts"):
# #             st.info(f"Retrieving {retmax} abstracts from PubMed...")
# #             retrieve_abstracts(st.session_state.ids[:retmax], email=email, db_path="abstracts.db")
# #             st.success("Abstracts saved to database ✅")
# #             st.session_state.retrieved = True

# #     # Analyze and visualize data if retrieved
# #     if st.session_state.retrieved:
# #         if st.button("📊 Analyze & Visualize"):
# #             df, cluster_titles, summaries = summarize_abstracts("abstracts.db")
# #             st.session_state.df = df
# #             st.session_state.summaries = summaries

# #         if st.session_state.df is not None:
# #             st.markdown("### 🧠 Clustered Abstracts")
# #             st.dataframe(st.session_state.df[['pmid', 'title', 'cluster']])

# #             st.image("pca_plot.png", caption="PCA Cluster Plot")
# #             st.image("tsne_plot.png", caption="t-SNE Cluster Plot")

# #             st.markdown("### 📝 Summary")
# #             for summary in st.session_state.summaries:
# #                 st.markdown(summary)

# #             with open("pubmed_review_results.zip", "rb") as f:
# #                 st.download_button("📥 Download ZIP", f, file_name="pubmed_review_results.zip")

# #     if st.button("📥 Download Raw Abstracts"):
# #         conn = sqlite3.connect("abstracts.db")
# #         df = pd.read_sql("SELECT * FROM abstracts", conn)
# #         conn.close()
# #         csv = df.to_csv(index=False)
# #         st.download_button("Download CSV", csv, file_name="abstracts.csv")

# # # # # import streamlit as st
# # # # # import sqlite3
# # # # # import bcrypt
# # # # # from utils import search_pubmed, retrieve_abstracts, summarize_abstracts
# # # # # import yaml
# # # # # from yaml.loader import SafeLoader
# # # # # import pandas as pd

# # # # # # Load config for default email or other info
# # # # # with open('config.yaml') as file:
# # # # #     config = yaml.load(file, Loader=SafeLoader)

# # # # # # --- Database helpers ---
# # # # # def create_users_table():
# # # # #     conn = sqlite3.connect('users.db')
# # # # #     c = conn.cursor()
# # # # #     c.execute("""
# # # # #         CREATE TABLE IF NOT EXISTS users (
# # # # #             username TEXT PRIMARY KEY,
# # # # #             password_hash BLOB,
# # # # #             email TEXT
# # # # #         )
# # # # #     """)
# # # # #     conn.commit()
# # # # #     conn.close()

# # # # # def add_user(username, password, email):
# # # # #     conn = sqlite3.connect('users.db')
# # # # #     c = conn.cursor()
# # # # #     hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
# # # # #     try:
# # # # #         c.execute("INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)", (username, hashed, email))
# # # # #         conn.commit()
# # # # #     except sqlite3.IntegrityError:
# # # # #         conn.close()
# # # # #         return False  # username already exists
# # # # #     conn.close()
# # # # #     return True

# # # # # def authenticate_user(username, password):
# # # # #     conn = sqlite3.connect('users.db')
# # # # #     c = conn.cursor()
# # # # #     c.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
# # # # #     result = c.fetchone()
# # # # #     conn.close()
# # # # #     if result:
# # # # #         stored_hash = result[0]
# # # # #         return bcrypt.checkpw(password.encode(), stored_hash)
# # # # #     return False

# # # # # # Create users table on app start
# # # # # create_users_table()

# # # # # # --- Streamlit UI ---
# # # # # st.title("🔐 PubMed Abstract Analyzer with Auth")

# # # # # # Authentication state stored in session state
# # # # # if 'logged_in' not in st.session_state:
# # # # #     st.session_state.logged_in = False
# # # # # if 'username' not in st.session_state:
# # # # #     st.session_state.username = ''

# # # # # menu = ["Login", "Sign up"]
# # # # # choice = st.sidebar.selectbox("Menu", menu)

# # # # # if choice == "Sign up":
# # # # #     st.subheader("Create New Account")
# # # # #     new_user = st.text_input("Username")
# # # # #     new_email = st.text_input("Email")
# # # # #     new_password = st.text_input("Password", type='password')
# # # # #     new_password_confirm = st.text_input("Confirm Password", type='password')
# # # # #     if st.button("Sign Up"):
# # # # #         if new_password != new_password_confirm:
# # # # #             st.error("Passwords do not match")
# # # # #         elif add_user(new_user, new_password, new_email):
# # # # #             st.success("Account created successfully. Please login.")
# # # # #         else:
# # # # #             st.error("Username already exists.")

# # # # # elif choice == "Login":
# # # # #     st.subheader("Login to your account")
# # # # #     username = st.text_input("Username")
# # # # #     password = st.text_input("Password", type='password')
# # # # #     if st.button("Login"):
# # # # #         if authenticate_user(username, password):
# # # # #             st.session_state.logged_in = True
# # # # #             st.session_state.username = username
# # # # #             st.success(f"Welcome {username}!")
# # # # #         else:
# # # # #             st.error("Invalid username or password")

# # # # # # Show logout and main app if logged in
# # # # # if st.session_state.logged_in:
# # # # #     if st.sidebar.button("Logout"):
# # # # #         st.session_state.logged_in = False
# # # # #         st.session_state.username = ''
# # # # #         st.experimental_rerun()

# # # # #     st.sidebar.success(f"Logged in as {st.session_state.username}")

# # # # #     with st.expander("🧠 Search Criteria Agreement", expanded=True):
# # # # #         st.write("By using this tool, you agree to use PubMed data responsibly and comply with NCBI API guidelines.")
# # # # #         st.markdown("""- Use your own email address for API identification.
# # # # # - Avoid excessive requests.
# # # # # - Results are for research purposes only.""")

# # # # #     email = st.text_input("Enter your email (used for NCBI API)", value=config.get('email', ''), key="email")
# # # # #     topic = st.text_input("🔍 Search Topic", "bone remodeling")
# # # # #     retmax = st.number_input("📄 Number of abstracts to retrieve", min_value=10, max_value=1000, value=100, step=10)
# # # # #     do_search = st.button("Start Search")

# # # # #     if do_search and email:
# # # # #         st.info("Searching PubMed...")
# # # # #         ids = search_pubmed(topic, email=email, retmax=retmax)
# # # # #         st.success(f"Found {len(ids)} abstracts.")

# # # # #         if ids:
# # # # #             st.info("Retrieving abstracts with sleep intervals...")
# # # # #             retrieve_abstracts(ids, email=email, db_path="abstracts.db")

# # # # #             st.success("Abstracts saved to database ✅")
# # # # #             if st.button("📊 Analyze & Visualize"):
# # # # #                 df, cluster_titles, summaries = summarize_abstracts("abstracts.db")
# # # # #                 st.markdown("### 🧠 Clustered Abstracts")
# # # # #                 st.dataframe(df[['pmid', 'title', 'cluster']])
# # # # #                 st.image("pca_plot.png", caption="PCA Cluster Plot")
# # # # #                  st.image("tsne_plot.png", caption="t-SNE Cluster Plot")
# # # # #                 st.markdown("### 📝 Summary")
# # # # #                 for summary in summaries:
# # # # #                 st.markdown(summary)
# # # # #                 with open("pubmed_review_results.zip", "rb") as f:
# # # # #                 st.download_button("📥 Download ZIP", f, file_name="pubmed_review_results.zip")

# # # # #             if st.button("Summarize Abstracts"):
# # # # #                 st.info("Generating summary...")
# # # # #                 summary = summarize_abstracts("abstracts.db")
# # # # #                 st.markdown("## 📝 Summary")
# # # # #                 st.markdown(summary)

# # # # #             if st.button("Download Abstracts"):
# # # # #                 conn = sqlite3.connect("abstracts.db")
# # # # #                 df = pd.read_sql("SELECT * FROM abstracts", conn)
# # # # #                 csv = df.to_csv(index=False)
# # # # #                 st.download_button("Download CSV", csv, file_name="abstracts.csv")
# # # # import streamlit as st
# # # # import sqlite3
# # # # import bcrypt
# # # # from utils import search_pubmed, retrieve_abstracts, summarize_abstracts
# # # # import yaml
# # # # from yaml.loader import SafeLoader
# # # # import pandas as pd

# # # # # Load config
# # # # with open('config.yaml') as file:
# # # #     config = yaml.load(file, Loader=SafeLoader)

# # # # # --- Auth DB ---
# # # # def create_users_table():
# # # #     conn = sqlite3.connect('users.db')
# # # #     c = conn.cursor()
# # # #     c.execute("""
# # # #         CREATE TABLE IF NOT EXISTS users (
# # # #             username TEXT PRIMARY KEY,
# # # #             password_hash BLOB,
# # # #             email TEXT
# # # #         )
# # # #     """)
# # # #     conn.commit()
# # # #     conn.close()

# # # # def add_user(username, password, email):
# # # #     conn = sqlite3.connect('users.db')
# # # #     c = conn.cursor()
# # # #     hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
# # # #     try:
# # # #         c.execute("INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)", (username, hashed, email))
# # # #         conn.commit()
# # # #     except sqlite3.IntegrityError:
# # # #         conn.close()
# # # #         return False
# # # #     conn.close()
# # # #     return True

# # # # def authenticate_user(username, password):
# # # #     conn = sqlite3.connect('users.db')
# # # #     c = conn.cursor()
# # # #     c.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
# # # #     result = c.fetchone()
# # # #     conn.close()
# # # #     if result:
# # # #         return bcrypt.checkpw(password.encode(), result[0])
# # # #     return False

# # # # create_users_table()

# # # # # --- Streamlit UI ---
# # # # st.title("🔐 PubMed Abstract Analyzer with Auth")

# # # # if 'logged_in' not in st.session_state:
# # # #     st.session_state.logged_in = False
# # # # if 'username' not in st.session_state:
# # # #     st.session_state.username = ''

# # # # menu = ["Login", "Sign up"]
# # # # choice = st.sidebar.selectbox("Menu", menu)

# # # # if choice == "Sign up":
# # # #     st.subheader("Create New Account")
# # # #     new_user = st.text_input("Username")
# # # #     new_email = st.text_input("Email")
# # # #     new_password = st.text_input("Password", type='password')
# # # #     new_password_confirm = st.text_input("Confirm Password", type='password')
# # # #     if st.button("Sign Up"):
# # # #         if new_password != new_password_confirm:
# # # #             st.error("Passwords do not match")
# # # #         elif add_user(new_user, new_password, new_email):
# # # #             st.success("Account created successfully. Please login.")
# # # #         else:
# # # #             st.error("Username already exists.")

# # # # elif choice == "Login":
# # # #     st.subheader("Login to your account")
# # # #     username = st.text_input("Username")
# # # #     password = st.text_input("Password", type='password')
# # # #     if st.button("Login"):
# # # #         if authenticate_user(username, password):
# # # #             st.session_state.logged_in = True
# # # #             st.session_state.username = username
# # # #             st.success(f"Welcome {username}!")
# # # #         else:
# # # #             st.error("Invalid username or password")

# # # # # --- Main App ---
# # # # if st.session_state.logged_in:
# # # #     if st.sidebar.button("Logout"):
# # # #         st.session_state.logged_in = False
# # # #         st.session_state.username = ''
# # # #         st.experimental_rerun()

# # # #     st.sidebar.success(f"Logged in as {st.session_state.username}")

# # # #     with st.expander("🧠 Search Criteria Agreement", expanded=True):
# # # #         st.write("By using this tool, you agree to use PubMed data responsibly and comply with NCBI API guidelines.")
# # # #         st.markdown("""- Use your own email address for API identification.
# # # # - Avoid excessive requests.
# # # # - Results are for research purposes only.""")

# # # #     email = st.text_input("Enter your email (used for NCBI API)", value=config.get('email', ''), key="email")
# # # #     topic = st.text_input("🔍 Search Topic", "bone remodeling")
# # # #     retmax = st.number_input("📄 Number of abstracts to retrieve", min_value=10, max_value=1000, value=100, step=10)
# # # #     do_search = st.button("Start Search")

# # # #     if do_search and email:
# # # #         st.info("Searching PubMed...")
# # # #         ids = search_pubmed(topic, email=email, retmax=retmax)
# # # #         st.success(f"Found {len(ids)} abstracts.")

# # # #         if ids:
# # # #             st.info("Retrieving abstracts from PubMed...")
# # # #             retrieve_abstracts(ids, email=email, db_path="abstracts.db")
# # # #             st.success("Abstracts saved to database ✅")

# # # #             if st.button("📊 Analyze & Visualize"):
# # # #                 df, cluster_titles, summaries = summarize_abstracts("abstracts.db")

# # # #                 st.markdown("### 🧠 Clustered Abstracts")
# # # #                 st.dataframe(df[['pmid', 'title', 'cluster']])

# # # #                 st.image("pca_plot.png", caption="PCA Cluster Plot")
# # # #                 st.image("tsne_plot.png", caption="t-SNE Cluster Plot")

# # # #                 st.markdown("### 📝 Summary")
# # # #                 for summary in summaries:
# # # #                     st.markdown(summary)

# # # #                 with open("pubmed_review_results.zip", "rb") as f:
# # # #                     st.download_button("📥 Download ZIP", f, file_name="pubmed_review_results.zip")

# # # #             if st.button("📥 Download Raw Abstracts"):
# # # #                 conn = sqlite3.connect("abstracts.db")
# # # #                 df = pd.read_sql("SELECT * FROM abstracts", conn)
# # # #                 conn.close()
# # # #                 csv = df.to_csv(index=False)
# # # #                 st.download_button("Download CSV", csv, file_name="abstracts.csv")
# # # import streamlit as st
# # # import sqlite3
# # # import bcrypt
# # # from utils import search_pubmed, retrieve_abstracts, summarize_abstracts
# # # import pandas as pd

# # # # --- Auth DB ---
# # # def create_users_table():
# # #     conn = sqlite3.connect('users.db')
# # #     c = conn.cursor()
# # #     c.execute("""
# # #         CREATE TABLE IF NOT EXISTS users (
# # #             username TEXT PRIMARY KEY,
# # #             password_hash BLOB,
# # #             email TEXT
# # #         )
# # #     """)
# # #     conn.commit()
# # #     conn.close()

# # # def add_user(username, password, email):
# # #     conn = sqlite3.connect('users.db')
# # #     c = conn.cursor()
# # #     hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
# # #     try:
# # #         c.execute("INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)", (username, hashed, email))
# # #         conn.commit()
# # #     except sqlite3.IntegrityError:
# # #         conn.close()
# # #         return False
# # #     conn.close()
# # #     return True

# # # def authenticate_user(username, password):
# # #     conn = sqlite3.connect('users.db')
# # #     c = conn.cursor()
# # #     c.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
# # #     result = c.fetchone()
# # #     conn.close()
# # #     if result:
# # #         return bcrypt.checkpw(password.encode(), result[0])
# # #     return False

# # # create_users_table()

# # # # --- Streamlit UI ---
# # # st.title("🔐 PubMed Abstract Analyzer with Auth")

# # # if 'logged_in' not in st.session_state:
# # #     st.session_state.logged_in = False
# # # if 'username' not in st.session_state:
# # #     st.session_state.username = ''

# # # menu = ["Login", "Sign up"]
# # # choice = st.sidebar.selectbox("Menu", menu)

# # # if choice == "Sign up":
# # #     st.subheader("Create New Account")
# # #     new_user = st.text_input("Username")
# # #     new_email = st.text_input("Email")
# # #     new_password = st.text_input("Password", type='password')
# # #     new_password_confirm = st.text_input("Confirm Password", type='password')
# # #     if st.button("Sign Up"):
# # #         if new_password != new_password_confirm:
# # #             st.error("Passwords do not match")
# # #         elif add_user(new_user, new_password, new_email):
# # #             st.success("Account created successfully. Please login.")
# # #         else:
# # #             st.error("Username already exists.")

# # # elif choice == "Login":
# # #     st.subheader("Login to your account")
# # #     username = st.text_input("Username")
# # #     password = st.text_input("Password", type='password')
# # #     if st.button("Login"):
# # #         if authenticate_user(username, password):
# # #             st.session_state.logged_in = True
# # #             st.session_state.username = username
# # #             st.success(f"Welcome {username}!")
# # #         else:
# # #             st.error("Invalid username or password")

# # # # --- Main App ---
# # # if st.session_state.logged_in:
# # #     if st.sidebar.button("Logout"):
# # #         st.session_state.logged_in = False
# # #         st.session_state.username = ''
# # #         st.experimental_rerun()

# # #     st.sidebar.success(f"Logged in as {st.session_state.username}")

# # #     with st.expander("🧠 Search Criteria Agreement", expanded=True):
# # #         st.write("By using this tool, you agree to use PubMed data responsibly and comply with NCBI API guidelines.")
# # #         st.markdown("""- Use your own email address for API identification.
# # # - Avoid excessive requests.
# # # - Results are for research purposes only.""")

# # #     email = st.text_input("Enter your email (used for NCBI API)")
# # #     topic = st.text_input("🔍 Search Topic", "bone remodeling")
# # #     do_search = st.button("Start Search")

# # #     if do_search and email and topic:
# # #         # Search for abstracts and get the total count
# # #         st.info("Searching PubMed...")
# # #         ids, total_count = search_pubmed(topic, email=email, retmax=100)

# # #         # Show the available count and let the user select the number of abstracts
# # #         st.success(f"Found {total_count} abstracts.")
# # #         retmax = st.slider("Number of abstracts to retrieve", min_value=10, max_value=total_count, value=100, step=10)

# # #         if retmax > 0:
# # #             st.info(f"Retrieving {retmax} abstracts from PubMed...")
# # #             retrieve_abstracts(ids[:retmax], email=email, db_path="abstracts.db")
# # #             st.success("Abstracts saved to database ✅")

# # #             if st.button("📊 Analyze & Visualize"):
# # #                 # Summarize abstracts and create visualizations
# # #                 df, cluster_titles, summaries = summarize_abstracts("abstracts.db")

# # #                 st.markdown("### 🧠 Clustered Abstracts")
# # #                 st.dataframe(df[['pmid', 'title', 'cluster']])

# # #                 st.image("pca_plot.png", caption="PCA Cluster Plot")
# # #                 st.image("tsne_plot.png", caption="t-SNE Cluster Plot")

# # #                 st.markdown("### 📝 Summary")
# # #                 for summary in summaries:
# # #                     st.markdown(summary)

# # #                 with open("pubmed_review_results.zip", "rb") as f:
# # #                     st.download_button("📥 Download ZIP", f, file_name="pubmed_review_results.zip")

# # #             if st.button("📥 Download Raw Abstracts"):
# # #                 conn = sqlite3.connect("abstracts.db")
# # #                 df = pd.read_sql("SELECT * FROM abstracts", conn)
# # #                 conn.close()
# # #                 csv = df.to_csv(index=False)
# # #                 st.download_button("Download CSV", csv, file_name="abstracts.csv")
