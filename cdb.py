import chromadb
import hashlib

# Initialize ChromaDB client
chroma_client = chromadb.Client()

# Create or get collection
collection = chroma_client.get_or_create_collection(name="my_collection")

# Function to generate a unique hash for a given text
def generate_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()

# Function to upsert documents with unique hash IDs
def upsert_documents(documents):
    ids = [generate_hash(doc) for doc in documents]
    collection.upsert(documents=documents, ids=ids)
    return ids

# Function to query the collection
def query_collection(query_texts, n_results):
    results = collection.query(query_texts=query_texts, n_results=n_results)
    return results

def document_exists(doc_id):
    try:
        existing_doc = chromadb_client.get(collection_name, doc_id)
        return existing_doc is not None
    except Exception:
        return False

