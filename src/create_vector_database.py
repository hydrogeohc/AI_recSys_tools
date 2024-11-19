from sentence_transformers import SentenceTransformer
import PyPDF2
import os
from transformers import pipeline
import json
import numpy as np
from datetime import datetime
from pymongo import MongoClient

# Initialize MongoDB client and collection
def init_mongodb():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['pdf_rag_database']
    collection = db['pdf_rag_embeddings']
    return collection

# Save the collection data to a JSON file
def save_db(collection, filename="mongodb_collection.json"):
    data = list(collection.find({}, {'_id': 0}))
    with open(filename, 'w') as file:
        json.dump(data, file)
    print(f"Database saved to {filename}")

# Load the collection data from a JSON file and recreate the collection
def load_db(filename="mongodb_collection.json"):
    collection = init_mongodb()
    with open(filename, 'r') as file:
        data = json.load(file)
    collection.insert_many(data)
    print(f"Database loaded from {filename}")
    return collection

# Extract text and split into chunks
def extract_and_chunk_text(file_path, chunk_size=500):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def process_pdfs_in_directory(directory, collection):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            chunks = extract_and_chunk_text(file_path)
            for i, chunk in enumerate(chunks):
                embedding = model.encode(chunk)
                timestamp = datetime.now().isoformat()
                collection.insert_one({
                    "embedding": embedding.tolist(),
                    "document": chunk,
                    "metadata": {
                        "source": filename,
                        "chunk_id": i,
                        "timestamp": timestamp
                    },
                    "id": f"{filename}_chunk_{i}"
                })
            print(f"Processed and stored: {filename}")

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
        {"$limit": 5}
    ])
    return list(results)

def generate_response(results, query):
    context = " ".join([result['document'] for result in results])
    generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")
    input_text = f"Context: {context}\nQuery: {query}\nAnswer:"
    response = generator(input_text, max_new_tokens=150, num_return_sequences=1)
    return response[0]['generated_text']

if __name__ == "__main__":
    pdf_directory = ".././PDF_dir/"
    collection = init_mongodb()
    
    # Uncomment the following line if you want to load an existing database
    # collection = load_db()
    
    process_pdfs_in_directory(pdf_directory, collection)
    save_db(collection)
    
    query = "What is the main topic of the PDFs?"
    results = query_database(collection, query)
    response = generate_response(results, query)
    print("Generated Response:", response)