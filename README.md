# UK Nuclear Regulations Chatbot

A GenAI-powered chatbot built to answer questions on UK nuclear safety, compliance, and regulations using a fine-tuned Retrieval-Augmented Generation (RAG) architecture. Developed during the #HackaFuture Hackathon hosted by AtkinsRéalis on February 2025.

---

## Overview

This chatbot leverages a **custom LLM (FLAN-T5-base)** integrated with **semantic search** and **vector embeddings (SBERT)** to extract and answer questions from real UK nuclear regulatory documents.

It’s specifically designed for:

- Regulatory compliance queries  
- Educational purposes for students and professionals  
- Legal and policy-based document Q&A  
- Safety guidance around nuclear operations in the UK  

---

## Features

- Uses **Retrieval-Augmented Generation (RAG)** to pull answers from PDF content  
- Based on **UK official regulatory documents**  
- Simple **Tkinter GUI** for full-screen interactive experience  
- Displays **source references** for every response  
- Handles True/False and reasoning-based prompts

## Directory Structure
UK_nuclear_chatbot/
├── pdf_files/              # UK nuclear regulatory PDF files
├── previous_attempts/      # Archived early prototypes
├── nuclear_chatbot 2.py    # Final working chatbot code with GUI
├── README.md               # This file
