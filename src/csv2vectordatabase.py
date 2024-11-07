import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import uuid

def load_csv(file_path):
    return pd.read_csv(file_path)

def process_csv(file_path):
    # Load CSV
    df = load_csv(file_path)
    
    # Initialize SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize ChromaDB client
    client = chromadb.Client()
    collection = client.create_collection("csv_data")

    # Process each row in the CSV
    for index, row in df.iterrows():
        # Combine all text columns into a single string
        text_content = ' '.join(row.astype(str))
        
        # Generate embedding
        embedding = model.encode(text_content)

        # Create a unique ID for each row
        unique_id = str(uuid.uuid4())

        # Store in ChromaDB
        collection.add(
            embeddings=[embedding.tolist()],
            documents=[text_content],
            metadatas=[row.to_dict()],
            ids=[unique_id]
        )
        print(f"Processed and stored row {index + 1}")

    return collection

def query_database(collection, query):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3
    )

    return results

# Main execution
if __name__ == "__main__":
    csv_file_path = "path/to/your/csv/file.csv"
    collection = process_csv(csv_file_path)

    # Example query
    query = "What is the main topic of the CSV data?"
    results = query_database(collection, query)
    print("Query Results:", results)

    # Optional: Persist the database
    persist_directory = "path/to/persist"
    persistent_client = chromadb.PersistentClient(path=persist_directory)
    persistent_collection = persistent_client.get_or_create_collection("csv_data")
    