# %%
!pip install chromadb langchain_community langchain_ollama langchain_core
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
# %%
# 1. Load documents from a local folder
#docs_path = "./source_texts/praxis_pietatis_verses"  # Change as needed
#loader = DirectoryLoader(docs_path, glob="*.json", loader_cls=JSONLoader)
#documents = loader.load()

# --- Add document splitting ---
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=0,
#     add_start_index=True,
#     separators=[
#         "\n\n",
#     ],
# )
# split_docs = text_splitter.split_documents(documents)
# len(split_docs)
# %%
pprint(documents)
# %%
# 2. Create embeddings using Ollama
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)  # Or another embedding model available in Ollama
# 3. Create Chroma vector store and ingest documents
# Open an existing Chroma collection from a persisted directory
#%%
persist_directory = "./chroma_db"
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
vectordb = Chroma.from_documents(
    documents,
    embeddings,
    persist_directory=persist_directory
)

# %%
# If you want to use the filter at retrieval time, Chroma supports metadata filtering via search_kwargs
retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3,
        # "fetch_k": 200,
        # "filter": {"source": {"$contains": "chapter1"}}
    },
)
# %%
docs = retriever.get_relevant_documents("ein gute wehr und waffen")
docs.sort(key=lambda x: x.metadata["source"])
docs

# %%
docs[8].page_content
