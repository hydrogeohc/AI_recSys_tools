import re
from PyPDF2 import PdfReader
from pymongo import MongoClient
import uuid
from sentence_transformers import SentenceTransformer
from datetime import datetime

# Initialize MongoDB client and collection
def init_mongodb(collection_name="keyword_extracted_sentences"):
    client = MongoClient('mongodb://localhost:27017/')
    db = client['your_database_name']
    collection = db[collection_name]
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
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Rest of the function remains the same
    # ...

# Main function to process PDF and store sentences
def process_pdf(pdf_path, keywords, client, collection):
    sentences = extract_sentences_from_pdf(pdf_path, keywords)
    
    # Use SentenceTransformer to generate embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    for sentence in sentences:
        embedding = model.encode(sentence).tolist()
        
        # Store in MongoDB
        document = {
            "id": str(uuid.uuid4()),
            "sentence": sentence,
            "embedding": embedding,
            "timestamp": datetime.now()
        }
        collection.insert_one(document)

# Usage
client, collection = init_mongodb()
process_pdf("your_pdf_file.pdf", ["keyword1", "keyword2"], client, collection)