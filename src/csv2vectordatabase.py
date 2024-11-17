import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
import json
import numpy as np
from datetime import datetime

# Initialize ChromaDB client and collection
def init_chromadb(collection_name="csv_data"):
    client = chromadb.Client()
    collection = client.create_collection(collection_name)
    return client, collection

# Utility function to convert numpy arrays to lists for JSON serialization
def convert_embeddings_to_serializable_format(embeddings):
    return [embedding.tolist() if isinstance(embedding, np.ndarray) else embedding for embedding in embeddings]

# Save the collection data to a JSON file
def save_db(collection, filename="chromadb_collection.json"):
    # Extract data to save, excluding 'ids'
    results = collection.get(include=["documents", "embeddings", "metadatas"])
    
    # Generate a list of unique IDs from the metadata
    ids = [str(uuid.uuid4()) for _ in range(len(results["documents"]))]
    results["ids"] = ids

    # Convert embeddings to a JSON-serializable format
    results["embeddings"] = convert_embeddings_to_serializable_format(results["embeddings"])
    
    # Save data to a JSON file
    with open(filename, 'w') as file:
        json.dump(results, file)
    print(f"Database saved to {filename}")

# Load the collection data from a JSON file and recreate the collection
def load_db(client, collection_name="csv_data", filename="chromadb_collection.json"):
    with open(filename, 'r') as file:
        data = json.load(file)
    
    collection = client.create_collection(collection_name)
    
    # Convert embeddings back to numpy arrays
    embeddings = [np.array(embedding) for embedding in data["embeddings"]]
    
    collection.add(
        embeddings=embeddings,
        documents=data["documents"],
        metadatas=data["metadatas"],
        ids=data["ids"]
    )
    print(f"Database loaded from {filename}")
    return collection

# Load CSV file
def load_csv(file_path):
    return pd.read_csv(file_path)

# Process CSV data and add to ChromaDB
def process_csv(file_path):
    # Load CSV
    df = load_csv(file_path)
    
    # Initialize SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize ChromaDB client and collection
    client, collection = init_chromadb()

    # Process each row in the CSV
    for index, row in df.iterrows():
        # Combine all text columns into a single string
        text_content = ' '.join(row.astype(str))
        
        # Generate embedding
        embedding = model.encode(text_content)

        # Create a unique ID for each row
        unique_id = str(uuid.uuid4())

        # Add a timestamp to the metadata
        timestamp = datetime.now().isoformat()

        # Store in ChromaDB
        collection.add(
            embeddings=[embedding.tolist()],
            documents=[text_content],
            metadatas=[{**row.to_dict(), "timestamp": timestamp}],
            ids=[unique_id]
        )
        print(f"Processed and stored row {index + 1}")

    return client, collection

# Query the database
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
    csv_file_path = ".././test_data.csv"
    client, collection = process_csv(csv_file_path)

    # Example query
    query = "What is the main topic of the CSV data?"
    results = query_database(collection, query)
    print("Query Results:", results)

    # Save the database to a file
    save_db(collection)


    # To load the database in a future session, uncomment the following:
    # loaded_collection = load_db(client)
