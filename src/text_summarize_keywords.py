import re
from PyPDF2 import PdfReader

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

# Example usage
pdf_path = "path/to/your/pdf/file.pdf"
keywords = ["natural language processing", "computer science", "artificial intelligence"]

extracted_content = extract_sentences_from_pdf(pdf_path, keywords, case_sensitive=False)

print("Extracted sentences:")
for sentence in extracted_content:
    print("- " + sentence)