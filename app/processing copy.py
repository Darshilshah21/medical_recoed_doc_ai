import pytesseract
from PIL import Image
import cv2
import os
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

INDEX_PATH = "vectorstore/index.faiss"
DATA_PATH = "vectorstore/data.json"

# ensure directory exists
os.makedirs("vectorstore", exist_ok=True)
load_dotenv()

client = OpenAI()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# embedding_model = SentenceTransformer("text-embedding-3-large")

INDEX_PATH = "vectorstore/index.faiss"
DATA_PATH = "vectorstore/data.json"

# Initialize Vector Storage
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
    with open(DATA_PATH, "r") as f:
        stored_docs = json.load(f)
else:
    # index = faiss.IndexFlatL2(384)
    index = faiss.IndexFlatL2(384)
    stored_docs = []

# Image Preprocessing
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # noise removal
    gray = cv2.medianBlur(gray, 3)
    # thresholding
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,2)
    # resize for better OCR
    scale_percent = 150
    width = int(thresh.shape[1] * scale_percent / 100)
    height = int(thresh.shape[0] * scale_percent / 100)
    resized = cv2.resize(thresh, (width, height))
    return resized

# OCR Extraction
def extract_text(image_path):
    processed = preprocess_image(image_path)
    config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(processed, config=config)
    return text.strip()

# Clean OCR text
def clean_text(text):
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if len(line) > 2:
            cleaned.append(line)
    return "\n".join(cleaned)

# Clinical Information Extraction
def extract_clinical_information(text):
    prompt = f"""
    You are a medical document parser.
    Extract clinical information from the medical document.
    Return STRICT JSON format only.
    Schema:
    {{
    "diagnoses": [],
    "medications": [],
    "lab_results": [],
    "allergies": [],
    "vitals": {{}}
    }}
    Medical Text:
    {text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return json.loads(response.choices[0].message.content)

# Vector Storage
def store_vector(text, structured_data):
    vector = embedding_model.encode([text])[0]
    vector = np.array([vector]).astype("float32")
    index.add(vector)
    stored_docs.append({
        "text": text,
        "structured": structured_data
    })
    faiss.write_index(index, INDEX_PATH)
    with open(DATA_PATH, "w") as f:
        json.dump(stored_docs, f, indent=2)

# Document Processing
def process_document(path):
    raw_text = extract_text(path)
    cleaned_text = clean_text(raw_text)
    clinical_info = extract_clinical_information(cleaned_text)
    store_vector(cleaned_text, clinical_info)
    return {
        "preview_text": cleaned_text[:500],
        "clinical_info": clinical_info
    }

# Search Documents
def search_docs(query):
    query_vector = embedding_model.encode([query])
    query_vector = np.array(query_vector).astype("float32")
    distances, indices = index.search(query_vector, 5)
    results = []
    for i in indices[0]:
        if i < len(stored_docs):
            results.append(stored_docs[i]["text"])
    return results

# Ask Question
def ask_question(question):
    context = search_docs(question)
    prompt = f"""
    You are a clinical AI assistant.
    Answer the question using ONLY the medical records context.
    Context:
    {context}
    Question:
    {question}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content