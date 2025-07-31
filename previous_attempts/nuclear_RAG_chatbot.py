from llama_parse import LlamaParse
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.kdbai import KDBAIVectorStore
from llama_index.postprocessor.cohere_rerank import CohereRerank
from getpass import getpass
import os
import kdbai_client as kdbai


LLAMA_CLOUD_API_KEY = 'llx-ENZJhDMjXLMWiSjwPRAwTlSokFtgWyOgcsVmSj6ChDDcQMpM'
OPENAI_API_KEY = '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'
KDBAI_ENDPOINT = 'https://cloud.kdb.ai/instance/ik2m7kqk35'
KDBAI_API_KEY = '075364370d-UAbb07S5NDDCrqzW0YgRL1BN4apOsgyvkGshozDlbHQcyjRSuNFDz2IEG136JDWHvI0Mqt+8R4DCgEiX'

#connect to KDB.AI
session = kdbai.Session(api_key=KDBAI_API_KEY, endpoint=KDBAI_ENDPOINT)

schema = [
        dict(name="document_id", type="str"),
        dict(name="text", type="str"),
        dict(name="embeddings", type="float32s"),
    ]

indexFlat = {
        "name": "flat",
        "type": "flat",
        "column": "embeddings",
        "params": {'dims': 1536, 'metric': 'L2'},
    }
# Connect with kdbai database
db = session.database("default")
KDBAI_TABLE_NAME = "LlamaParse_Table"

# check if table already exists
existing_tables = [i.name for i in db.tables]
if not KDBAI_TABLE_NAME in existing_tables:
    # create the table
    table = db.create_table(KDBAI_TABLE_NAME, schema, indexes=[indexFlat])

EMBEDDING_MODEL  = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4o"

llm = OpenAI(model=GENERATION_MODEL)
embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)

Settings.llm = llm
Settings.embed_model = embed_model
pdf_file_name = '../data_source/Compliance with the Convention on Nuclear Safety Obligations_ 9th National Report - GOV.UK.pdf'
parsing_instructions = '''
The document titled "A Guide to Nuclear Regulation in the UK" is a regulatory guide published by the **Office for Nuclear Regulation (ONR)**. It provides a comprehensive overview of nuclear regulation in the UK, covering the **legal framework, safety, security, emergency planning, international regulations, and nuclear safeguards**.

The document outlines:
- **The role of ONR** in regulating nuclear safety, security, and transportation of radioactive materials.
- **UK nuclear industry structure**, including operational reactors, decommissioning sites, and fuel processing facilities.
- **Safety and risk management**, including regulatory compliance measures, site inspections, and emergency preparedness.
- **Security regulations**, such as the Nuclear Industries Security Regulations (NISR) 2003, focusing on protecting nuclear sites and materials.
- **Nuclear emergency planning**, detailing response frameworks for different levels of nuclear incidents.
- **International nuclear regulations and safeguards**, including compliance with the **IAEA, Euratom, and the Nuclear Non-Proliferation Treaty (NPT)**.

The document also includes detailed tables listing **UK nuclear sites, reactor types, decommissioning plans, and regulatory conditions**. It is a critical resource for understanding **nuclear regulatory compliance** in the UK.

When answering questions using this document:
- **Base responses on the official regulatory guidelines and principles outlined.**
- **Refer to ONR's role in nuclear safety and security oversight.**
- **If a question is about specific legislation, refer to relevant sections on the Nuclear Installations Act 1965, the Energy Act 2013, or other applicable laws.**
- **For safety-related inquiries, reference ONR’s risk management framework, emergency preparedness plans, and defense-in-depth safety strategies.**
- **Provide accurate and concise regulatory explanations.**
'''

# Initialize LlamaParse with API Key
llama_parser = LlamaParse(
    api_key=LLAMA_CLOUD_API_KEY,
    result_type="markdown",
    parsing_instructions=parsing_instructions
)

# Load PDF document
documents = llama_parser.load_data(pdf_file_name)

# parse the documents using MarkdownElementNodeParser
node_parser = MarkdownElementNodeParser(llm=llm, num_workers=8).from_defaults()
# retrieve nodes (text) and objects (table)
nodes = node_parser.get_nodes_from_documents(documents)
base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
# insert the table markdown into the text of each table object
for i in range(len(objects)):
  objects[i].text = objects[i].obj.text[:]
print(objects[0])
vector_store = KDBAIVectorStore(table)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
# create the index, inserts base_nodes and objects into KDB.AI
recursive_index = VectorStoreIndex(
    nodes= base_nodes + objects, storage_context=storage_context
)
# Query KDB.AI to ensure the nodes were inserted
print(table.query())

from openai import OpenAI
client = OpenAI()


def embed_query(query):
    query_embedding = client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
    return query_embedding.data[0].embedding


def retrieve_data(query):
    query_embedding = embed_query(query)
    results = table.search(vectors={'flat':[query_embedding]}, n=5,
                           filter=[('<>','document_id','4a9551df-5dec-4410-90bb-43d17d722918')])
    retrieved_data_for_RAG = []
    for index, row in results[0].iterrows():
      retrieved_data_for_RAG.append(row['text'])
    return retrieved_data_for_RAG


def RAG(query):
    system_prompt = ("You are a Nuclear Regulation and Compliance Assistant, responsible for "
                     "providing accurate and authoritative answers strictly based on UK nuclear "
                     "regulatory documents. Your responses must align with laws, safety protocols, and "
                     "compliance measures as outlined by the Office for Nuclear Regulation (ONR), "
                     "the IAEA, and UK legislation. If relevant information is found in the "
                     "reference material, use it exclusively. If no relevant context is available, "
                     "respond by stating that the provided documents do not "
                     "contain the required information.")

    question = "You will answer this question based on the provided reference material: " + query
    messages = "Here is the provided context: " + "\n"

    results = retrieve_data(query)  # Retrieves relevant regulatory excerpts

    if results:
        for data in results:
            messages += data + "\n"
    else:
        messages += "No relevant regulatory information was found in the provided reference material."

    # Debugging: Print retrieved context before sending to OpenAI
    print("\n--- Retrieved Context for Query ---")
    print(messages[:1000])  # Print first 1000 chars of retrieved content

    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{messages}\n\nQuestion: {query}"}
        ],
        max_tokens=300,
    )

    content = response.choices[0].message.content
    return content

print(RAG(input("Please Input your query here!!")))
