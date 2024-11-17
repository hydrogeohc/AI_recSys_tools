import re
from PyPDF2 import PdfReader
import chromadb
import uuid
from sentence_transformers import SentenceTransformer
from datetime import datetime

# Initialize ChromaDB client and collection
def init_chromadb(collection_name="keyword_extracted_sentences"):
    client = chromadb.Client()
    collection = client.create_collection(collection_name)
    return client, collection

# Extract sentences from PDF that match given keywords
def extract_sentences_from_pdf(pdf_path, keywords, case_sensitive=False):
    # Read the PDF file
    reader = PdfReader(pdf_path)
    
    # Extract text from all pages
    text = ""
    for page in reader.pages:
        text += page.extract_text() + " "
    
    # Split the text into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    
    # Convert keywords to lowercase if case-insensitive
    if not case_sensitive:
        keywords = [keyword.lower() for keyword in keywords]
    
    extracted_sentences = []
    
    for sentence in sentences:
        # Convert sentence to lowercase if case-insensitive
        sentence_to_check = sentence if case_sensitive else sentence.lower()
        
        # Check if any keyword is in the sentence
        if any(keyword in sentence_to_check for keyword in keywords):
            extracted_sentences.append(sentence.strip())
    
    return extracted_sentences

# Process extracted sentences and add them to ChromaDB
def add_sentences_to_chromadb(pdf_path, sentences, collection):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    for i, sentence in enumerate(sentences):
        # Generate embedding for the sentence
        embedding = model.encode(sentence)
        
        # Create a unique ID for each sentence
        unique_id = f"{pdf_path}_sentence_{i}"
        
        # Add a timestamp to the metadata
        timestamp = datetime.now().isoformat()

        # Add to ChromaDB
        collection.add(
            embeddings=[embedding.tolist()],
            documents=[sentence],
            metadatas=[{"source": pdf_path, "sentence_id": i, "timestamp": timestamp}],
            ids=[unique_id]
        )
        print(f"Processed and stored sentence {i + 1} from {pdf_path}")

# Main execution
if __name__ == "__main__":
    pdf_path = "../PDF_dir/2411.04578v1.pdf"
    keywords = ["natural language processing", "computer science", "artificial intelligence"]

    # Extract sentences containing the keywords
    extracted_content = extract_sentences_from_pdf(pdf_path, keywords, case_sensitive=False)

    if extracted_content:
        print("Extracted sentences:")
        for sentence in extracted_content:
            print("- " + sentence)
        
        # Initialize ChromaDB client and collection
        client, collection = init_chromadb()

        # Add extracted sentences to ChromaDB
        add_sentences_to_chromadb(pdf_path, extracted_content, collection)

        # Optional: Save the database to a file (use save_db function if needed)
        # save_db(collection)

    else:
        print("No sentences found with the specified keywords.")
