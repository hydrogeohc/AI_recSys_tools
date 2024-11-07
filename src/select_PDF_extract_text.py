import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
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

# Usage example
pdf_path = "path/to/your/pdf/file.pdf"
extracted_text = extract_text_from_pdf(pdf_path)

# Print or save the extracted text
print(extracted_text)

# Optionally, save to a text file
with open("extracted_text.txt", "w", encoding="utf-8") as text_file:
    text_file.write(extracted_text)