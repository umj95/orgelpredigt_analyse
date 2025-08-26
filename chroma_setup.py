# %%
from typing import Iterable, List
from langchain_community.document_loaders import DirectoryLoader, TextLoader, JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pprint import pprint
import os
import json

# %%
# Read all verses for each songbook page into a document list
documents = []
for i in range(41,1291):
    with open(f"source_texts/praxis_pietatis_verses/{i}.json") as f:
        data = json.load(f)
    
    for sentence in data:
        document = Document(
            page_content = sentence,
            metadata = {'source': f'{i}'}
        )
        documents.append(document)

# %%
pprint(documents)
# %%
# 2. Create embeddings using Ollama
model = "nomic-embed-text"
embeddings = OllamaEmbeddings(
    model=model
)  # Or another embedding model available in Ollama
# 3. Create Chroma vector store and ingest documents
persist_directory = f"./chroma/chroma_db_{model}"
# Open an existing Chroma collection from a persisted directory
#%%
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
vectordb = Chroma.from_documents(
    documents,
    embeddings,
    persist_directory=persist_directory
)