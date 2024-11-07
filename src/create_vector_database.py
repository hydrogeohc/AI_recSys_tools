import PyPDF2
from sentence_transformers import SentenceTransformer
import chromadb
import os

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def process_pdfs_in_directory(directory):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.Client()
    collection = client.create_collection("pdf_embeddings")

    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            pdf_text = extract_text_from_pdf(file_path)
            
            # Generate embedding
            embedding = model.encode(pdf_text)

            # Store in ChromaDB
            collection.add(
                embeddings=[embedding.tolist()],
                documents=[pdf_text],
                metadatas=[{"source": filename}],
                ids=[filename]
            )
            print(f"Processed and stored: {filename}")

    return collection

def query_database(collection, query):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=1
    )

    return results

# Main execution
if __name__ == "__main__":
    pdf_directory = "path/to/your/pdf/directory"
    collection = process_pdfs_in_directory(pdf_directory)

    # Example query
    query = "What is the main topic of the PDFs?"
    results = query_database(collection, query)
    print("Query Results:", results)