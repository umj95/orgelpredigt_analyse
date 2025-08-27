# %%
from typing import Iterable, List
#from langchain_community.document_loaders import DirectoryLoader, TextLoader, JSONLoader
from langchain_community.vectorstores import Chroma
#from langchain_ollama import OllamaEmbeddings
from langchain_core.documents.base import Document
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from pprint import pprint
#import os
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import json
from sentence_transformers import SentenceTransformer

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
# 2. Create embeddings
#model = "nomic-embed-text"
model_name = "LaBSE"
model = SentenceTransformer(f'sentence-transformers/{model_name}')
#%%
#embeddings = model.encode([doc.page_content for doc in test])

#embeddings = OllamaEmbeddings(
#    model=model
#)
#%%
from langchain_core.embeddings.embeddings import Embeddings
class EmbedSomething(Embeddings):
    def __init__(self,model) -> None:
        self.model = model

    def embed_documents(self,texts):
        t = self.model.encode(texts)
        return t.tolist()

    def embed_query(self, text: str) -> List[float]:
        t = self.model.encode(text)
        return t.tolist()


emb = EmbedSomething(model)
#embeddings
#%%
# 3. Create Chroma vector store and ingest documents
persist_directory = f"./chroma/chroma_db_{model_name}"
# Open an existing Chroma collection from a persisted directory
#%%
vectordb = Chroma(persist_directory=persist_directory, embedding_function=  emb)
vectordb_store = Chroma.from_documents(
    documents,
    emb,
    persist_directory=persist_directory
)
# %%
#vectordb_store.search("der erst teil", search_type="similarity")
vectordb_store.similarity_search_with_score("erste theil dieses büchleins")

# %%
