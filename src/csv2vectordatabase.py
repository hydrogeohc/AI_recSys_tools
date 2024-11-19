import pandas as pd
from sentence_transformers import SentenceTransformer
import uuid
import json
import numpy as np
from datetime import datetime
from pymongo import MongoClient

# Initialize MongoDB client and collection
def init_mongodb(collection_name="csv_data"):
    client = MongoClient('mongodb://localhost:27017/')
    db = client['vector_database']
    collection = db[collection_name]
    return client, collection

# Utility function to convert numpy arrays to lists for JSON serialization
def convert_embeddings_to_serializable_format(embeddings):
    return [embedding.tolist() if isinstance(embedding, np.ndarray) else embedding for embedding in embeddings]

# Save the collection data to a JSON file
def save_db(collection, filename="mongodb_collection.json"):
    results = list(collection.find({}, {'_id': 0}))
    
    # Convert embeddings to a JSON-serializable format
    for doc in results:
        doc['embedding'] = convert_embeddings_to_serializable_format([doc['embedding']])[0]
    
    # Save data to a JSON file
    with open(filename, 'w') as file:
        json.dump(results, file)
    print(f"Database saved to {filename}")

# Load the collection data from a JSON file and recreate the collection
def load_db(client, collection_name="csv_data", filename="mongodb_collection.json"):
    db = client['vector_database']
    collection = db[collection_name]
    
    with open(filename, 'r') as file:
        data = json.load(file)
    
    # Convert embeddings back to numpy arrays and insert into MongoDB
    for doc in data:
        doc['embedding'] = np.array(doc['embedding'])
        collection.insert_one(doc)
    
    print(f"Database loaded from {filename}")
    return collection

# Load CSV file
def load_csv(file_path):
    return pd.read_csv(file_path)

# Process CSV data and add to MongoDB
def process_csv(file_path):
    # Load CSV
    df = load_csv(file_path)
    
    # Initialize SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize MongoDB client and collection
    client, collection = init_mongodb()
    
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
        
        # Store in MongoDB
        document = {
            'id': unique_id,
            'text_content': text_content,
            'embedding': embedding.tolist(),
            'metadata': {**row.to_dict(), "timestamp": timestamp}
        }
        collection.insert_one(document)
        
        print(f"Processed and stored row {index + 1}")
    
    return client, collection

# Query the database
def query_database(collection, query):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query)
    
    # Perform a similarity search using dot product
    results = collection.aggregate([
        {
            '$addFields': {
                'similarity': {
                    '$reduce': {
                        'input': {'$zip': {'inputs': ['$embedding', query_embedding.tolist()]}},
                        'initialValue': 0,
                        'in': {'$add': ['$$value', {'$multiply': ['$$this.0', '$$this.1']}]}
                    }
                }
            }
        },
        {'$sort': {'similarity': -1}},
        {'$limit': 3},
        {'$project': {'_id': 0, 'text_content': 1, 'metadata': 1, 'similarity': 1}}
    ])
    
    return list(results)

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