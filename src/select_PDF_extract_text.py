import pandas as pd
from sentence_transformers import SentenceTransformer
import uuid
import json
import numpy as np
import fitz # PyMuPDF
from datetime import datetime
from pymongo import MongoClient

# Initialize MongoDB client and collection
def init_mongodb(db_name="vector_db", collection_name="data_collection"):
    client = MongoClient('mongodb://localhost:27017/')
    db = client[db_name]
    collection = db[collection_name]
    return client, collection

# Utility function to convert numpy arrays to lists for JSON serialization
def convert_embeddings_to_serializable_format(embeddings):
    return [embedding.tolist() if isinstance(embedding, np.ndarray) else embedding for embedding in embeddings]

# Save the collection data to a JSON file
def save_db(collection, filename="mongodb_collection.json"):
    data = list(collection.find({}, {'_id': 0}))
    for item in data:
        item['embedding'] = convert_embeddings_to_serializable_format(item['embedding'])
    
    with open(filename, 'w') as file:
        json.dump(data, file)
    print(f"Database saved to {filename}")

# Load the collection data from a JSON file and recreate the collection
def load_db(client, db_name="vector_db", collection_name="data_collection", filename="mongodb_collection.json"):
    with open(filename, 'r') as file:
        data = json.load(file)
    
    db = client[db_name]
    collection = db[collection_name]
    collection.drop()  # Clear existing data
    
    for item in data:
        item['embedding'] = np.array(item['embedding'])
        collection.insert_one(item)
    
    print(f"Database loaded from {filename}")
    return collection

# Extract text from PDF using PyMuPDF
def extract_text_from_pdf(pdf_path):
    try:
        pdf_document = fitz.open(pdf_path)
        all_text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text = page.get_text()
            all_text += f"\n\n--- Page {page_num + 1} ---\n\n{text}"
        pdf_document.close()
        return all_text
    except Exception as e:
        print(f"Error while extracting text from PDF: {e}")
        return ""

# Process PDF data and add to MongoDB
def process_pdf_to_mongodb(pdf_path, collection):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    text_content = extract_text_from_pdf(pdf_path)
    
    if not text_content:
        print(f"No text extracted from {pdf_path}. Skipping.")
        return
    
    chunk_size = 500
    chunks = [text_content[i:i+chunk_size] for i in range(0, len(text_content), chunk_size)]
    
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk)
        unique_id = f"{pdf_path}_chunk_{i}"
        timestamp = datetime.now().isoformat()
        
        document = {
            "_id": unique_id,
            "text": chunk,
            "embedding": embedding.tolist(),
            "metadata": {
                "source": pdf_path,
                "chunk_id": i,
                "timestamp": timestamp
            }
        }
        collection.insert_one(document)
        print(f"Processed and stored chunk {i + 1} from {pdf_path}")

# Load CSV file
def load_csv(file_path):
    return pd.read_csv(file_path)

# Process CSV data and add to MongoDB
def process_csv(file_path, collection):
    df = load_csv(file_path)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    for index, row in df.iterrows():
        text_content = ' '.join(row.astype(str))
        embedding = model.encode(text_content)
        unique_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        document = {
            "_id": unique_id,
            "text": text_content,
            "embedding": embedding.tolist(),
            "metadata": {**row.to_dict(), "timestamp": timestamp}
        }
        collection.insert_one(document)
        print(f"Processed and stored row {index + 1}")

# Query the database
def query_database(collection, query):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query)
    
    results = collection.aggregate([
        {
            "$addFields": {
                "similarity": {
                    "$dotProduct": ["$embedding", query_embedding.tolist()]
                }
            }
        },
        {"$sort": {"similarity": -1}},
        {"$limit": 3},
        {"$project": {"_id": 0, "text": 1, "metadata": 1, "similarity": 1}}
    ])
    
    return list(results)

# Main execution
if __name__ == "__main__":
    client, collection = init_mongodb()
    
    # Process CSV and PDF files
    csv_file_path = ".././test_data.csv"
    pdf_file_path = "../PDF_dir/2411.04578v1.pdf"
    process_csv(csv_file_path, collection)
    process_pdf_to_mongodb(pdf_file_path, collection)
    
    # Example query
    query = "What is the main topic of the data in the database?"
    results = query_database(collection, query)
    print("Query Results:", results)
    
    # Save the database to a file
    save_db(collection)
    
    # To load the database in a future session, uncomment the following:
    # loaded_collection = load_db(client)