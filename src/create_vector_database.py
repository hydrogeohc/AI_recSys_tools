from sentence_transformers import SentenceTransformer
import PyPDF2
import chromadb
import os
from transformers import pipeline
import json
import numpy as np

# Initialize ChromaDB client and collection
def init_chromadb():
    client = chromadb.Client()
    collection = client.create_collection("pdf_rag_embeddings")
    return collection

# Utility function to convert numpy arrays to lists for JSON serialization
def convert_embeddings_to_serializable_format(embeddings):
    return [embedding.tolist() if isinstance(embedding, np.ndarray) else embedding for embedding in embeddings]

# Save the collection data to a JSON file
def save_db(collection, filename="chromadb_collection.json"):
    # Extract data to save, excluding 'ids'
    results = collection.get(include=["documents", "embeddings", "metadatas"])
    
    # Generate a list of unique IDs from the metadata
    ids = [metadata.get('chunk_id') for metadata in results["metadatas"]]
    results["ids"] = ids

    # Convert embeddings to a JSON-serializable format
    results["embeddings"] = convert_embeddings_to_serializable_format(results["embeddings"])
    
    # Save data to a JSON file
    with open(filename, 'w') as file:
        json.dump(results, file)
    print(f"Database saved to {filename}")

# Load the collection data from a JSON file and recreate the collection
def load_db(client, collection_name="pdf_rag_embeddings", filename="chromadb_collection.json"):
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

# Extract text and split into chunks
def extract_and_chunk_text(file_path, chunk_size=500):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

    # Split text into chunks of defined size
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def process_pdfs_in_directory(directory, collection):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            chunks = extract_and_chunk_text(file_path)

            for i, chunk in enumerate(chunks):
                # Generate embedding for each chunk
                embedding = model.encode(chunk)

                # Store in ChromaDB
                collection.add(
                    embeddings=[embedding.tolist()],
                    documents=[chunk],
                    metadatas=[{"source": filename, "chunk_id": i}],
                    ids=[f"{filename}_chunk_{i}"]
                )
            print(f"Processed and stored: {filename}")

def query_database(collection, query):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query)

    # Retrieve top N relevant chunks
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5
    )
    return results

# Revised function to handle response generation
def generate_response(results, query):
    print("Results structure:", results)  # Debug line to verify structure

    # Check if the results structure is as expected
    if 'documents' in results and isinstance(results['documents'][0], list):
        context = " ".join(results['documents'][0])  # Combine the retrieved chunks into context
    else:
        raise ValueError("Unexpected results structure")

    # Initialize the text generation pipeline with a valid model
    generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")  # Ensure the model is valid

    # Pass context and query to the generative model
    input_text = f"Context: {context}\nQuery: {query}\nAnswer:"
    response = generator(input_text, max_new_tokens=150, num_return_sequences=1)  # Changed max_length to max_new_tokens
    return response[0]['generated_text']

# Main execution
if __name__ == "__main__":
    pdf_directory = ".././PDF_dir/"
    client = chromadb.Client()
    
    # Uncomment the following line if you want to load an existing database
    # collection = load_db(client)

    collection = init_chromadb()
    process_pdfs_in_directory(pdf_directory, collection)

    # Save the database after processing
    save_db(collection)

    # Example query
    query = "What is the main topic of the PDFs?"
    results = query_database(collection, query)
    response = generate_response(results, query)
    print("Generated Response:", response)
