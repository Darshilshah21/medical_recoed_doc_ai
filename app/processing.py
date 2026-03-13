import pytesseract
from PIL import Image
import cv2
import os
import faiss
import json
import numpy as np
import re

from sentence_transformers import SentenceTransformer
from openai import OpenAI
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

INDEX_PATH = "vectorstore/index.faiss"
DATA_PATH = "vectorstore/data.json"

os.makedirs("vectorstore", exist_ok=True)
os.makedirs("data/patients", exist_ok=True)


# Load Vector DB
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
    with open(DATA_PATH) as f:
        stored_docs = json.load(f)
else:
    index = faiss.IndexFlatL2(384)
    stored_docs = []

# Extract Patient ID
def extract_patient_id(filename):
    match = re.search(r"[A-Z]{3}-\d{4}-[A-Z]{2}-\d+", filename)
    if match:
        return match.group(0)
    return "UNKNOWN_PATIENT"

# Image preprocessing
def preprocess_image(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,2)
    return thresh

# OCR
def extract_text(path):
    processed = preprocess_image(path)
    config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(processed, config=config)
    return text.strip()

# Clean OCR
def clean_text(text):
    lines = text.split("\n")
    cleaned = []
    for l in lines:
        l = l.strip()
        if len(l) > 2:
            cleaned.append(l)
    return "\n".join(cleaned)

# LLM Extraction
def extract_clinical_information(text):
    prompt = f"""
    You are a medical document parser.
    Extract clinical information.
    Return JSON:
    {{
    "diagnoses": [],
    "medications": [],
    "lab_results": [],
    "allergies": [],
    "vitals": {{}}
    }}

    TEXT: {text}
"""
    # openai

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type":"json_object"},
        messages=[{"role":"user","content":prompt}]
    )
    return json.loads(response.choices[0].message.content)

    # gemini

    # response = client_gemini.models.generate_content(
    #     model="gemini-2.0-flash-lite",
    #     contents=[
    #         {
    #             "role":"user",
    #             "parts":[{"text":prompt}]
    #         }
    #     ]
    # )
    # output = response.text
    # output = output.replace("```json","").replace("```","")
    # try:
    #     return json.loads(output)
    # except:
    #     return {
    #         "diagnoses": [],
    #         "medications": [],
    #         "lab_results": [],
    #         "allergies": [],
    #         "vitals": {}
    #     }


# Store patient record
def store_patient_record(patient_id, text):
    patient_file = f"data/patients/{patient_id}.txt"
    if os.path.exists(patient_file):
        with open(patient_file,"a",encoding="utf8") as f:
            f.write("\n"+text)
    else:
        with open(patient_file,"w",encoding="utf8") as f:
            f.write(text)

# Store vector
def store_vector(patient_id, text):
    vector = embedding_model.encode([text])[0]
    vector = np.array([vector]).astype("float32")
    index.add(vector)
    stored_docs.append({
        "patient_id":patient_id,
        "text":text
    })
    faiss.write_index(index, INDEX_PATH)
    with open(DATA_PATH,"w") as f:
        json.dump(stored_docs,f,indent=2)

# Process document
def process_document(path):
    filename = os.path.basename(path)
    patient_id = extract_patient_id(filename)
    raw = extract_text(path)
    clean = clean_text(raw)
    clinical_info = extract_clinical_information(clean)
    store_patient_record(patient_id, clean)
    store_vector(patient_id, clean)
    return {
        "patient_id":patient_id,
        "preview":clean[:400],
        "clinical_info":clinical_info
    }


# Get patient list
def get_patient_list():
    files = os.listdir("data/patients")
    ids = [f.replace(".txt","") for f in files]
    return ids

# Search within patient
def search_docs(query, patient_id):
    query_vector = embedding_model.encode([query])
    query_vector = np.array(query_vector).astype("float32")
    D,I = index.search(query_vector,10)
    results = []
    for idx in I[0]:
        if idx < len(stored_docs):
            doc = stored_docs[idx]
            if doc["patient_id"] == patient_id:
                results.append(doc["text"])
    return results[:5]


# Ask question
def ask_question(question, patient_id):
    context = search_docs(question, patient_id)
    prompt = f"""
    You are a clinical AI assistant.
    Answer using ONLY this patient record.
    Patient ID: {patient_id}
    Context: {context}
    Question: {question}
"""
    # opanai

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )

    return response.choices[0].message.content

    # gemini

    # response = client_gemini.models.generate_content(
    #     model="gemini-2.0-flash-lite",
    #     contents=[
    #         {
    #             "role":"user",
    #             "parts":[{"text":prompt}]
    #         }
    #     ]
    # )
    # return response.text
