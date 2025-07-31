import kdbai_client as kdbai
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration

# API Keys
KDBAI_ENDPOINT = 'https://cloud.kdb.ai/instance/ik2m7kqk35'
KDBAI_API_KEY = '075364370d-UAbb07S5NDDCrqzW0YgRL1BN4apOsgyvkGshozDlbHQcyjRSuNFDz2IEG136JDWHvI0Mqt+8R4DCgEiX'
KDBAI_TABLE_NAME = "LlamaParse_Table"

# Connect to KDB.AI
session = kdbai.Session(api_key=KDBAI_API_KEY, endpoint=KDBAI_ENDPOINT)
db = session.database("default")
table = db.table(KDBAI_TABLE_NAME)

# Function to embed query
def embed_query(query):
    embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    embedding = embed_model.encode(query).tolist()
    return embedding


# Retrieve relevant data
def retrieve_data(query):
    query_embedding = embed_query(query)
    results = table.search(vectors={'flat': [query_embedding]}, n=5)

    retrieved_data_for_RAG = []
    for index, row in results[0].iterrows():
        retrieved_data_for_RAG.append(row['text'])

    # Debugging: Print the retrieved documents
    # print("\nRetrieved Documents:\n", retrieved_data_for_RAG)

    return retrieved_data_for_RAG


# RAG Function
# Load the model and tokenizer
model_name = "content/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


def RAG(query):
    system_prompt = (
        "You are a Nuclear Regulation and Compliance Assistant, responsible for providing accurate answers "
        "strictly based on UK nuclear regulatory documents. If no relevant information is found, state that "
        "no regulatory data is available."
    )

    results = retrieve_data(query)

    if not results:
        return "No relevant regulatory information was found in the provided reference material."

    # Select the two most relevant sentences
    sorted_results = sorted(results, key=len, reverse=True)
    selected_context = ". ".join(sorted_results[:4])

    # Add "UK-specific" instructions in the prompt
    input_text = (
        f"{system_prompt}\n\n"
        f"Context (UK Nuclear Regulation and Energy Research):\n{selected_context}\n\n"
        f"Provide a UK-specific answer to the following question:\n"
        f"Question: {query}\n\n"
    )

    # Debugging: Print the final input sent to FLAN-T5
    # print("\nFINAL INPUT TO MODEL \n", input_text)

    # Tokenize and generate response
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=100,
            num_beams=5,
            temperature=0.1,
            repetition_penalty=2.0,
            do_sample=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# User Query Input
query = input("Enter your nuclear regulation question: ")
print(RAG(query))
