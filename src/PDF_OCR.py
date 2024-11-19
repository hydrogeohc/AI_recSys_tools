import pandas as pd
from sentence_transformers import SentenceTransformer
import uuid
import json
import numpy as np
import fitz # PyMuPDF
import pytesseract
import cv2
from pdf2image import convert_from_path
import os
from datetime import datetime
from pymongo import MongoClient

# Initialize MongoDB client and collection
def init_mongodb(db_name="pdf_database", collection_name="data_collection"):
    client = MongoClient('mongodb://localhost:27017/')
    db = client[db_name]
    collection = db[collection_name]
    return client, collection

# Save the collection data to a JSON file
def save_db(collection, filename="mongodb_collection.json"):
    data = list(collection.find({}, {'_id': 0}))
    with open(filename, 'w') as file:
        json.dump(data, file)
    print(f"Database saved to {filename}")

# Load the collection data from a JSON file and recreate the collection
def load_db(client, db_name="pdf_database", collection_name="data_collection", filename="mongodb_collection.json"):
    db = client[db_name]
    collection = db[collection_name]
    
    with open(filename, 'r') as file:
        data = json.load(file)
    
    collection.insert_many(data)
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
            "embedding": embedding.tolist(),
            "text": chunk,
            "metadata": {
                "source": pdf_path,
                "chunk_id": i,
                "timestamp": timestamp
            }
        }
        
        collection.insert_one(document)
        print(f"Processed and stored chunk {i + 1} from {pdf_path}")

# Detect tables and figures in PDF pages using OpenCV
def detect_tables_and_figures_in_pdf(pdf_path, collection):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    pages = convert_from_path(pdf_path)
    
    for i, page in enumerate(pages):
        image = np.array(page)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        ocr_text = pytesseract.image_to_string(image)
        unique_id = f"{pdf_path}_page_{i}_ocr"
        timestamp = datetime.now().isoformat()
        
        if ocr_text.strip():
            embedding = model.encode(ocr_text)
            document = {
                "_id": unique_id,
                "embedding": embedding.tolist(),
                "text": ocr_text,
                "metadata": {
                    "source": pdf_path,
                    "page_num": i,
                    "content_type": "OCR",
                    "timestamp": timestamp
                }
            }
            
            collection.insert_one(document)
            print(f"Processed and stored OCR text for page {i + 1}")

# Main execution
if __name__ == "__main__":
    client, collection = init_mongodb()
    pdf_file_path = "../PDF_dir/2411.04578v1.pdf"
    
    process_pdf_to_mongodb(pdf_file_path, collection)
    detect_tables_and_figures_in_pdf(pdf_file_path, collection)
    
    # Save the database to a file
    save_db(collection)
    
    # To load the database in a future session, uncomment the following:
    # loaded_collection = load_db(client)
    
    # Close the MongoDB connection
    client.close()