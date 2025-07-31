import os
from sentence_transformers import SentenceTransformer
import pandas as pd
import kdbai_client as kdbai
import urllib.parse
import requests
from bs4 import BeautifulSoup

# API Keys
KDBAI_ENDPOINT = ''
KDBAI_API_KEY = ''

# Connect to KDB.AI
session = kdbai.Session(api_key=KDBAI_API_KEY, endpoint=KDBAI_ENDPOINT)
db = session.database("default")
KDBAI_TABLE_NAME = "LlamaParse_Table"

# Check if table exists
existing_tables = [i.name for i in db.tables]
if KDBAI_TABLE_NAME in existing_tables:
    print(f"Table '{KDBAI_TABLE_NAME}' already exists. Skipping table creation.")
    table = db.table(KDBAI_TABLE_NAME)
else:
    print(f"Table '{KDBAI_TABLE_NAME}' does not exist. Creating it now...")
    schema = [
        dict(name="document_id", type="str"),
        dict(name="title", type="str"),
        dict(name="text", type="str"),
        dict(name="embeddings", type="float32s"),
    ]
    indexFlat = {
        "name": "flat",
        "type": "flat",
        "column": "embeddings",
        "params": {'dims': 768, 'metric': 'L2'},
    }
    table = db.create_table(KDBAI_TABLE_NAME, schema, indexes=[indexFlat])
    print(f"Table '{KDBAI_TABLE_NAME}' created successfully.")

wiki_topics = [
    {"topic": "Office for Nuclear Regulation", "text": ""},
    {"topic": "Nuclear safety regulations", "text": ""},
    {"topic": "Nuclear power in the United Kingdom", "text": ""},
    {"topic": "United Kingdom Atomic Energy Authority", "text": ""},
    {"topic": "EDF Energy", "text": ""},
    {"topic": "Advanced gas-cooled reactor", "text": ""},
    {"topic": "Nuclear Decommissioning Authority", "text": ""},
    {"topic": "Civil Nuclear Constabulary", "text": ""}
]

for topic in wiki_topics:
    url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(topic['topic'])}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    content_div = soup.find("div", {"class": "mw-parser-output"})
    exclude_sections = {"See also", "References", "External links"}
    extracted_text = []
    for element in content_div.find_all(["h2", "h3", "p"]):
        if element.name in ["h2", "h3"]:
            section_title = element.get_text().strip()
            if any(section_title.startswith(exclude) for exclude in exclude_sections):
                break
        elif element.name == "p":
            extracted_text.append(element.get_text())
    extracted_text = ''.join(extracted_text)
    topic['text'] = extracted_text
    print(topic['topic'])

print(f"Loaded {len(wiki_topics)} articles from Wikipedia.")

embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Prepare data for insertion
data_to_insert = []
for i, item in enumerate(wiki_topics):
    embedding = embed_model.encode(item["text"]).tolist()
    data_to_insert.append({
        "document_id": str(i),
        "title": item["topic"],
        "text": item["text"],
        "embeddings": embedding
    })
    print(f"embeddings done for {item["topic"]}")
    print("Embedding Dimension:", len(embedding))

df = pd.DataFrame(data_to_insert)
table.insert(df)
print("Inserted Wikipedia embeddings into KDB.AI successfully.")
