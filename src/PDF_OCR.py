import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
import json
import numpy as np
import fitz  # PyMuPDF
import pytesseract
import cv2
from pdf2image import convert_from_path
import os

# Initialize ChromaDB client and collection
def init_chromadb(collection_name="data_collection"):
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
def load_db(client, collection_name="data_collection", filename="chromadb_collection.json"):
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

# Extract text from PDF using PyMuPDF
def extract_text_from_pdf(pdf_path):
    try:
        # Open the PDF file
        pdf_document = fitz.open(pdf_path)
        
        # Initialize an empty string to store all extracted text
        all_text = ""
        
        # Loop through each page
        for page_num in range(len(pdf_document)):
            # Get the page
            page = pdf_document[page_num]
            
            # Extract text from the page
            text = page.get_text()
            
            # Add page number and extracted text to the result
            all_text += f"\n\n--- Page {page_num + 1} ---\n\n{text}"
        
        # Close the PDF document
        pdf_document.close()
        
        return all_text
    except Exception as e:
        print(f"Error while extracting text from PDF: {e}")
        return ""

# Process PDF data and add to ChromaDB
def process_pdf_to_chromadb(pdf_path, collection):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Extract text from the PDF
    text_content = extract_text_from_pdf(pdf_path)
    if not text_content:
        print(f"No text extracted from {pdf_path}. Skipping.")
        return

    # Split the extracted text into chunks (optional step)
    chunk_size = 500  # You can modify this based on your needs
    chunks = [text_content[i:i+chunk_size] for i in range(0, len(text_content), chunk_size)]

    # Process each chunk and add to ChromaDB
    for i, chunk in enumerate(chunks):
        # Generate embedding
        embedding = model.encode(chunk)

        # Create a unique ID for each chunk
        unique_id = f"{pdf_path}_chunk_{i}"

        # Store in ChromaDB
        collection.add(
            embeddings=[embedding.tolist()],
            documents=[chunk],
            metadatas=[{"source": pdf_path, "chunk_id": i}],
            ids=[unique_id]
        )
        print(f"Processed and stored chunk {i + 1} from {pdf_path}")

# Detect tables and figures in PDF pages using OpenCV
def detect_tables_and_figures_in_pdf(pdf_path, collection):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    pages = convert_from_path(pdf_path)
    
    for i, page in enumerate(pages):
        # Convert PDF page to image
        image = np.array(page)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Detect tables and figures
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Detect lines for tables
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Detect contours for figures
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Perform OCR to extract text from the page
        ocr_text = pytesseract.image_to_string(image)
        unique_id = f"{pdf_path}_page_{i}_ocr"

        # Add OCR text and metadata to ChromaDB
        if ocr_text.strip():
            embedding = model.encode(ocr_text)
            collection.add(
                embeddings=[embedding.tolist()],
                documents=[ocr_text],
                metadatas=[{"source": pdf_path, "page_num": i, "content_type": "OCR"}],
                ids=[unique_id]
            )
            print(f"Processed and stored OCR text for page {i + 1}")

# Main execution
if __name__ == "__main__":
    client, collection = init_chromadb()


    pdf_file_path = "../PDF_dir/2411.04578v1.pdf"
    
    process_pdf_to_chromadb(pdf_file_path, collection)
    detect_tables_and_figures_in_pdf(pdf_file_path, collection)

    # Save the database to a file
    save_db(collection)

    # To load the database in a future session, uncomment the following:
    # loaded_collection = load_db(client)


