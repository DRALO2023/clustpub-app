import time
import sqlite3
from Bio import Entrez

def search_pubmed(term, email, retmax=100):
    Entrez.email = email
    handle = Entrez.esearch(db="pubmed", term=term, retmax=retmax)
    record = Entrez.read(handle)
    return record["IdList"]

def retrieve_abstracts(id_list, email, db_path="abstracts.db"):
    Entrez.email = email
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS abstracts (pmid TEXT, title TEXT, abstract TEXT)")

    for pmid in id_list:
        try:
            handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="xml")
            records = Entrez.read(handle)
            for article in records['PubmedArticle']:
                try:
                    title = article['MedlineCitation']['Article']['ArticleTitle']
                    abstract = article['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
                    cursor.execute("INSERT INTO abstracts VALUES (?, ?, ?)", (pmid, title, abstract))
                except:
                    continue
            conn.commit()
            time.sleep(0.4)  # Respect API rate
        except:
            continue
    conn.close()

def summarize_abstracts(db_path="abstracts.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT abstract FROM abstracts")
    abstracts = cursor.fetchall()
    conn.close()

    full_text = " ".join([a[0] for a in abstracts if a[0]])
    return f"🧾 Summary of {len(abstracts)} abstracts:

{full_text[:1000]}..."