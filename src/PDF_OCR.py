import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_path
import os

# Function to detect tables
def detect_tables(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return image

# Function to detect figures
def detect_figures(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    return image

# Main process
def extract_content(pdf_path):
    pages = convert_from_path(pdf_path)
    
    for i, page in enumerate(pages):
        # Convert PDF page to image
        image = np.array(page)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Detect tables and figures
        image_with_tables = detect_tables(image.copy())
        image_with_figures = detect_figures(image.copy())
        
        # Perform OCR
        text = pytesseract.image_to_string(image)
        
        # Save results
        cv2.imwrite(f'page_{i+1}_tables.png', image_with_tables)
        cv2.imwrite(f'page_{i+1}_figures.png', image_with_figures)
        
        with open(f'page_{i+1}_text.txt', 'w', encoding='utf-8') as f:
            f.write(text)

# Path to your PDF file
pdf_path = './*.pdf'
extract_content(pdf_path)
