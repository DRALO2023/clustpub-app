import sqlite3
import time
import pandas as pd
from Bio import Entrez
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

def search_pubmed(topic, email, retmax=100):
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

def retrieve_abstracts(id_list, email, db_path="abstracts.db", batch_size=100):
    Entrez.email = email
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS abstracts (
        pmid TEXT PRIMARY KEY, title TEXT, abstract TEXT)""")
    for start in range(0, len(id_list), batch_size):
        end = min(start + batch_size, len(id_list))
        ids = id_list[start:end]
        handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="xml")
        papers = Entrez.read(handle)
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

    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    X = tfidf.fit_transform(df['abstract'])
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)

    feature_names = tfidf.get_feature_names_out()
    cluster_titles = {}
    for clus in range(5):
        cluster_docs = X[df['cluster'] == clus]
        mean = cluster_docs.mean(axis=0).A1
        top_words = [feature_names[i] for i in mean.argsort()[::-1][:3]]
        cluster_titles[clus] = ", ".join(top_words)

    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    summaries = []
    for clus in sorted(df['cluster'].unique()):
        clus_df = df[df['cluster'] == clus]
        text = " ".join(clus_df['abstract'].tolist())[:3000]
        summary = summarizer(text, max_length=512, min_length=200, do_sample=False)[0]['summary_text']
        summaries.append(f"## Topic {clus + 1}: {cluster_titles[clus]}\n\n{summary}\n\n")

    df.to_csv("pubmed_abstracts_clusters.csv", index=False)

    with open("literature_review.md", "w", encoding="utf-8") as f:
        f.write("# Literature Review\n\n" + "\n".join(summaries))

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X.toarray())
    df_plot = pd.DataFrame(reduced, columns=["x", "y"])
    df_plot['cluster'] = df['cluster'].map(cluster_titles)
    plt.figure()
    sns.scatterplot(data=df_plot, x='x', y='y', hue='cluster')
    plt.title("PCA Cluster Plot")
    plt.savefig("pca_plot.png")

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_tsne = tsne.fit_transform(X.toarray())
    df_plot_tsne = pd.DataFrame(reduced_tsne, columns=["x", "y"])
    df_plot_tsne['cluster'] = df['cluster'].map(cluster_titles)
    plt.figure()
    sns.scatterplot(data=df_plot_tsne, x='x', y='y', hue='cluster')
    plt.title("t-SNE Cluster Plot")
    plt.savefig("tsne_plot.png")

    with zipfile.ZipFile("pubmed_review_results.zip", "w") as zipf:
        zipf.write("pubmed_abstracts_clusters.csv")
        zipf.write("literature_review.md")
        zipf.write("pca_plot.png")
        zipf.write("tsne_plot.png")

    return df, cluster_titles, summaries

# import sqlite3
# import time
# import pandas as pd
# from Bio import Entrez
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from transformers import pipeline
# import matplotlib.pyplot.pyplot as plt
# import seaborn as sns
# import zipfile

# def search_pubmed(topic, email, retmax=100):
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

# def retrieve_abstracts(id_list, email, db_path="abstracts.db", batch_size=100):
#     Entrez.email = email
#     conn = sqlite3.connect(db_path)
#     c = conn.cursor()
#     c.execute("""
#         CREATE TABLE IF NOT EXISTS abstracts (
#             pmid TEXT PRIMARY KEY,
#             title TEXT,
#             abstract TEXT
#         )
#     """)
#     for start in range(0, len(id_list), batch_size):
#         end = min(start + batch_size, len(id_list))
#         ids = id_list[start:end]
#         handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="xml")
#         papers = Entrez.read(handle)
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

#     tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
#     X = tfidf.fit_transform(df['abstract'])
#     kmeans = KMeans(n_clusters=5, random_state=42)
#     df['cluster'] = kmeans.fit_predict(X)

#     feature_names = tfidf.get_feature_names_out()
#     cluster_titles = {}
#     for clus in range(5):
#         cluster_docs = X[df['cluster'] == clus]
#         mean = cluster_docs.mean(axis=0).A1
#         top_words = [feature_names[i] for i in mean.argsort()[::-1][:3]]
#         cluster_titles[clus] = ", ".join(top_words)

#     summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
#     summaries = []
#     for clus in sorted(df['cluster'].unique()):
#         clus_df = df[df['cluster'] == clus]
#         text = " ".join(clus_df['abstract'].tolist())[:3000]
#         summary = summarizer(text, max_length=512, min_length=200, do_sample=False)[0]['summary_text']
#         summaries.append(f"## Topic {clus + 1}: {cluster_titles[clus]}\n\n{summary}\n\n")

#     df.to_csv("pubmed_abstracts_clusters.csv", index=False)

#     with open("literature_review.md", "w", encoding="utf-8") as f:
#         f.write("# Literature Review\n\n" + "\n".join(summaries))

#     pca = PCA(n_components=2)
#     reduced = pca.fit_transform(X.toarray())
#     df_plot = pd.DataFrame(reduced, columns=["x", "y"])
#     df_plot['cluster'] = df['cluster'].map(cluster_titles)
#     plt.figure()
#     sns.scatterplot(data=df_plot, x='x', y='y', hue='cluster')
#     plt.title("PCA Cluster Plot")
#     plt.savefig("pca_plot.png")

#     tsne = TSNE(n_components=2, perplexity=30, random_state=42)
#     reduced_tsne = tsne.fit_transform(X.toarray())
#     df_plot_tsne = pd.DataFrame(reduced_tsne, columns=["x", "y"])
#     df_plot_tsne['cluster'] = df['cluster'].map(cluster_titles)
#     plt.figure()
#     sns.scatterplot(data=df_plot_tsne, x='x', y='y', hue='cluster')
#     plt.title("t-SNE Cluster Plot")
#     plt.savefig("tsne_plot.png")

#     with zipfile.ZipFile("pubmed_review_results.zip", "w") as zipf:
#         zipf.write("pubmed_abstracts_clusters.csv")
#         zipf.write("literature_review.md")
#         zipf.write("pca_plot.png")
#         zipf.write("tsne_plot.png")

#     return df, cluster_titles, summaries

# # # import time
# # # import sqlite3
# # # from Bio import Entrez

# # # def search_pubmed(term, email, retmax=100):
# # #     Entrez.email = email
# # #     handle = Entrez.esearch(db="pubmed", term=term, retmax=retmax)
# # #     record = Entrez.read(handle)
# # #     return record["IdList"]

# # # def retrieve_abstracts(id_list, email, db_path="abstracts.db"):
# # #     Entrez.email = email
# # #     conn = sqlite3.connect(db_path)
# # #     cursor = conn.cursor()
# # #     cursor.execute("CREATE TABLE IF NOT EXISTS abstracts (pmid TEXT, title TEXT, abstract TEXT)")

# # #     for pmid in id_list:
# # #         try:
# # #             handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="xml")
# # #             records = Entrez.read(handle)
# # #             for article in records['PubmedArticle']:
# # #                 try:
# # #                     title = article['MedlineCitation']['Article']['ArticleTitle']
# # #                     abstract = article['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
# # #                     cursor.execute("INSERT INTO abstracts VALUES (?, ?, ?)", (pmid, title, abstract))
# # #                 except:
# # #                     continue
# # #             conn.commit()
# # #             time.sleep(0.4)  # Respect API rate
# # #         except:
# # #             continue
# # #     conn.close()

# # # def summarize_abstracts(db_path="abstracts.db"):
# # #     conn = sqlite3.connect(db_path)
# # #     cursor = conn.cursor()
# # #     cursor.execute("SELECT abstract FROM abstracts")
# # #     abstracts = cursor.fetchall()
# # #     conn.close()

# # #     full_text = " ".join([a[0] for a in abstracts if a[0]])
# # #     return f"""🧾 Summary of {len(abstracts)} abstracts:

# # # {full_text[:1000]}..."""
# # import sqlite3
# # import time
# # import pandas as pd
# # from Bio import Entrez
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.cluster import KMeans
# # from sklearn.decomposition import PCA
# # from sklearn.manifold import TSNE
# # from transformers import pipeline
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # import zipfile

# # from Bio import Entrez

# # def search_pubmed(topic, email, retmax=100):
# #     """
# #     Search PubMed for a given topic and return a list of PubMed IDs.
# #     Optionally, allow dynamic `retmax` based on total available abstracts.
# #     """
# #     Entrez.email = email
    
# #     # Perform the initial search to get the total count
# #     handle = Entrez.esearch(db="pubmed", term=topic, retmax=1)
# #     record = Entrez.read(handle)
# #     handle.close()
    
# #     # Get the total number of results
# #     total_count = int(record["Count"])
    
# #     # Set retmax to the total count or user-defined retmax
# #     retmax = min(retmax, total_count)  # Don't exceed the total available count
    
# #     # Now fetch the results with the correct `retmax`
# #     handle = Entrez.esearch(db="pubmed", term=topic, retmax=retmax)
# #     record = Entrez.read(handle)
# #     handle.close()
    
# #     return record['IdList'], total_count  # Return the IDs and total count


# # def retrieve_abstracts(id_list, email, db_path="abstracts.db", batch_size=100):
# #     Entrez.email = email
# #     conn = sqlite3.connect(db_path)
# #     c = conn.cursor()
# #     c.execute("""CREATE TABLE IF NOT EXISTS abstracts (
# #         pmid TEXT PRIMARY KEY, title TEXT, abstract TEXT)""")

# #     for start in range(0, len(id_list), batch_size):
# #         end = min(start + batch_size, len(id_list))
# #         ids = id_list[start:end]
# #         handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="xml")
# #         papers = Entrez.read(handle)
# #         for paper in papers.get('PubmedArticle', []):
# #             try:
# #                 pmid = paper['MedlineCitation']['PMID']
# #                 title = paper['MedlineCitation']['Article']['ArticleTitle']
# #                 abstract = paper['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
# #                 c.execute("INSERT OR IGNORE INTO abstracts VALUES (?, ?, ?)", (pmid, title, abstract))
# #             except Exception:
# #                 continue
# #         conn.commit()
# #         time.sleep(0.4)
# #     conn.close()

# # def summarize_abstracts(db_path="abstracts.db"):
# #     conn = sqlite3.connect(db_path)
# #     df = pd.read_sql("SELECT * FROM abstracts", conn)
# #     conn.close()

# #     tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
# #     X = tfidf.fit_transform(df['abstract'])
# #     kmeans = KMeans(n_clusters=5, random_state=42)
# #     df['cluster'] = kmeans.fit_predict(X)

# #     feature_names = tfidf.get_feature_names_out()
# #     cluster_titles = {}
# #     for clus in range(5):
# #         cluster_docs = X[df['cluster'] == clus]
# #         mean = cluster_docs.mean(axis=0).A1
# #         top_words = [feature_names[i] for i in mean.argsort()[::-1][:3]]
# #         cluster_titles[clus] = ", ".join(top_words)

# #     summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
# #     summaries = []
# #     for clus in sorted(df['cluster'].unique()):
# #         clus_df = df[df['cluster'] == clus]
# #         text = " ".join(clus_df['abstract'].tolist())[:3000]
# #         summary = summarizer(text, max_length=512, min_length=200, do_sample=False)[0]['summary_text']
# #         summaries.append(f"## Topic {clus + 1}: {cluster_titles[clus]}\n\n{summary}\n\n")

# #     # Save results
# #     df.to_csv("pubmed_abstracts_clusters.csv", index=False)

# #     with open("literature_review.md", "w", encoding="utf-8") as f:
# #         f.write("# Literature Review\n\n" + "\n".join(summaries))

# #     # PCA Plot
# #     pca = PCA(n_components=2)
# #     reduced = pca.fit_transform(X.toarray())
# #     df_plot = pd.DataFrame(reduced, columns=["x", "y"])
# #     df_plot['cluster'] = df['cluster'].map(cluster_titles)
# #     plt.figure()
# #     sns.scatterplot(data=df_plot, x='x', y='y', hue='cluster')
# #     plt.title("PCA Cluster Plot")
# #     plt.savefig("pca_plot.png")

# #     # t-SNE Plot
# #     tsne = TSNE(n_components=2, perplexity=30, random_state=42)
# #     reduced_tsne = tsne.fit_transform(X.toarray())
# #     df_plot_tsne = pd.DataFrame(reduced_tsne, columns=["x", "y"])
# #     df_plot_tsne['cluster'] = df['cluster'].map(cluster_titles)
# #     plt.figure()
# #     sns.scatterplot(data=df_plot_tsne, x='x', y='y', hue='cluster')
# #     plt.title("t-SNE Cluster Plot")
# #     plt.savefig("tsne_plot.png")

# #     # Create ZIP
# #     with zipfile.ZipFile("pubmed_review_results.zip", "w") as zipf:
# #         zipf.write("pubmed_abstracts_clusters.csv")
# #         zipf.write("literature_review.md")
# #         zipf.write("pca_plot.png")
# #         zipf.write("tsne_plot.png")

# #     return df, cluster_titles, summaries

