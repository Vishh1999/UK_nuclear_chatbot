from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer
import torch
import tkinter as tk
from tkinter import scrolledtext

EMB_SBERT_MPNET_BASE = "sentence-transformers/all-mpnet-base-v2"
LLM_FLAN_T5_BASE = "content/flan-t5-base"

config = {"persist_directory": None,
          "load_in_8bit": False,
          "embedding" : EMB_SBERT_MPNET_BASE,
          "llm": LLM_FLAN_T5_BASE,
          }


def create_sbert_mpnet():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(model_name=EMB_SBERT_MPNET_BASE, model_kwargs={"device": device})


def create_flan_t5_base(load_in_8bit=False):
    # Wrap it in HF pipeline for use with LangChain
    model = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model)
    return pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300, # 150
        model_kwargs={"device_map": "auto", "load_in_8bit": load_in_8bit, "max_length": 768,
                      "temperature": 0.3, "do_sample": True} # 512, 0.
    )


if config["embedding"] == EMB_SBERT_MPNET_BASE:
    embedding = create_sbert_mpnet()
load_in_8bit = config["load_in_8bit"]
if config["llm"] == LLM_FLAN_T5_BASE:
    llm = create_flan_t5_base(load_in_8bit=load_in_8bit)

# Load the pdf
pdf_path = "pdf_files/guide_to_nuclear_regulation_uk.pdf"
loader = PDFPlumberLoader(pdf_path)
documents = loader.load()
print("pdf is loaded")

# Split documents and create text snippets
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0) # 100, 0
texts = text_splitter.split_documents(documents)
text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=10, encoding_name="cl100k_base") # 1000, 10
texts = text_splitter.split_documents(texts)
print("pdf split is done")
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

text_splitter = TokenTextSplitter(chunk_size=1200, chunk_overlap=100, encoding_name="cl100k_base")
texts = text_splitter.split_documents(texts)
persist_directory = config["persist_directory"]
vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)

hf_llm = HuggingFacePipeline(pipeline=llm)
retriever = vectordb.as_retriever(search_kwargs={"k": 4}) # 4
qa = RetrievalQA.from_chain_type(llm=hf_llm, chain_type="stuff",retriever=retriever)

# Defining a default prompt for the llm
question_t5_template = """
You are a Nuclear Regulation and Compliance Assistant, responsible for providing accurate answers 
strictly based on UK nuclear regulatory documents. If no relevant information is found, state that 
no regulatory data is available.
context: {context}
question: {question}
answer: 
"""
QUESTION_T5_PROMPT = PromptTemplate(
    template=question_t5_template, input_variables=["context", "question"]
)
qa.combine_documents_chain.llm_chain.prompt = QUESTION_T5_PROMPT
qa.combine_documents_chain.verbose = True
qa.return_source_documents = True

# while True:
#     question = input("Enter your question: ")
#     if question == 'EXIT':
#         print("Exiting the Application. Thank you")
#         break
#     response = qa({"query": question})
#     print("Answer:\n", response["result"])
#     print("\nSource Documents:\n", response["source_documents"])

# Function to process user query
def get_response():
    user_query = user_input.get().strip()
    if not user_query:
        return  # Don't process empty queries

    # Enable chat area for writing
    chat_area.config(state=tk.NORMAL)

    # Add user query to chat history
    chat_area.insert(tk.END, f"\n\nUser: {user_query}\n", "user")
    chat_area.yview(tk.END)  # Scroll down

    if user_query.upper() == "EXIT":
        chat_area.insert(tk.END, "\nBot: Exiting the Application. Thank you!\n", "bot")
        chat_area.yview(tk.END)
        root.after(2000, root.destroy)  # Close app after 2 seconds
        return

    # Get response from model
    response = qa({"query": user_query})
    bot_answer = response["result"]

    # Display bot response
    chat_area.insert(tk.END, f"\nBot: {bot_answer}\n", "bot")
    chat_area.yview(tk.END)  # Scroll down

    # Display source documents (if available)
    if "source_documents" in response:
        source_texts = "\n".join([doc.page_content[:300] + "..." for doc in response["source_documents"]])
        chat_area.insert(tk.END, f"\nSource Documents:\n{source_texts}\n", "source")
        chat_area.yview(tk.END)

    # Disable chat area to prevent manual edits
    chat_area.config(state=tk.DISABLED)

    # Clear user input
    user_input.delete(0, tk.END)

# Setup Tkinter UI
root = tk.Tk()
root.title("Nuclear Regulation Chatbot")
root.configure(bg="white")

# Make the window full screen
root.attributes("-fullscreen", True)

# Heading Label
heading_label = tk.Label(
    root, text="Nuclear Regulation Chatbot", font=("Arial", 24, "bold"), bg="darkblue", fg="white", pady=10
)
heading_label.pack(fill=tk.X)

# Description Label
description_label = tk.Label(
    root, text="Ask about UK nuclear regulations, compliance, and safety policies. Type your question below!",
    font=("Arial", 14), fg="black", pady=5
)
description_label.pack()

# Chat Display Area
chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=20, font=("Arial", 12))
chat_area.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
chat_area.tag_configure("user", foreground="blue", font=("Arial", 12, "bold"))
chat_area.tag_configure("bot", foreground="green", font=("Arial", 12, "italic"))
chat_area.tag_configure("source", foreground="gray", font=("Arial", 10, "italic"))
chat_area.config(state=tk.DISABLED)  # Initially read-only

# User Input Field
user_input = tk.Entry(root, width=80, font=("Arial", 12))
user_input.pack(padx=20, pady=10)

# Buttons Frame
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

# Submit Button
submit_btn = tk.Button(button_frame, text="Ask", command=get_response, font=("Arial", 12, "bold"), bg="lightblue")
submit_btn.pack(side=tk.LEFT, padx=10)

# Exit Button
exit_btn = tk.Button(button_frame, text="Exit", command=root.destroy, font=("Arial", 12, "bold"), bg="lightblue")
exit_btn.pack(side=tk.RIGHT, padx=10)

# Run the Tkinter Loop
root.mainloop()
