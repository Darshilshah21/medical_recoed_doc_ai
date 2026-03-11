# Medical Records DocAI

## Overview

Hospitals receive many types of medical documents such as lab reports, discharge summaries, and billing pages. These documents are often scanned and difficult to quickly interpret.

This project implements a prototype **Medical Records DocAI system** that allows clinicians to:

* Upload patient medical documents
* Automatically extract key clinical information
* Search patient records
* Ask questions about the patient’s medical history

The goal is to demonstrate how **OCR, vector search, and LLMs** can help structure and query unstructured medical records.

---

# Problem Understanding

Medical documents are typically:

* Scanned images
* Poorly structured
* Spread across multiple pages
* Hard to search

This system attempts to solve this by:

1. Extracting text using OCR
2. Structuring clinical information using an LLM
3. Storing document embeddings in a vector database
4. Allowing semantic search and question answering

---

# Key Features

## 1. Upload Medical Documents

Users can upload scanned medical documents through the web interface.

Supported pipeline:

Upload → OCR → Cleaning → Information Extraction → Vector Storage

---

## 2. Extract Clinical Information

The system extracts key information from documents:

* Diagnoses
* Medications
* Lab Results
* Allergies
* Vitals

This extraction is performed using **Google Gemini LLM**.

---

## 3. Patient Identification

Patient names may be redacted in documents.

The system instead uses the **Patient ID embedded in the filename**.

Example:

AHD-0425-PA-0007719.png

This ID is used to group documents belonging to the same patient.

---

## 4. Search Patient Records

Users can search within a patient's documents.

The system uses:

* **SentenceTransformers embeddings**
* **FAISS vector database**

This allows **semantic search** across OCR-extracted content.

---

## 5. Ask Questions About a Patient

Users can ask natural language questions like:

* "What medications is the patient taking?"
* "Does the patient have any allergies?"

The system retrieves relevant records and uses an **LLM to generate answers grounded in the patient data**.

---

# System Architecture

```
User Upload
     ↓
OCR Processing (Tesseract)
     ↓
Text Cleaning
     ↓
Clinical Information Extraction (Gemini LLM)
     ↓
Embedding Generation (SentenceTransformers)
     ↓
Vector Storage (FAISS)
     ↓
Search / Question Answering
```

---

# Technology Stack

Backend

* Python
* Flask

AI / NLP

* Google Gemini API
* SentenceTransformers
* FAISS Vector Database

OCR

* Tesseract OCR
* OpenCV

Frontend

* HTML
* JavaScript
* Simple CSS

---

# Project Structure

```
medical_docai
│
├── app
│   ├── routes.py
│   ├── processing.py
│   ├── templates
│   │    └── index.html
│   └── static
│        └── js
│             └── main.js
│
├── data
│   ├── uploads
│   └── patients
│
├── vectorstore
│
├── main.py
└── requirements.txt
```

---

# Running the Project

## 1 Install Dependencies

```
pip install -r requirements.txt
```

## 2 Install Tesseract OCR

Download:

https://github.com/UB-Mannheim/tesseract/wiki

Update path if needed.

---

## 3 Set Environment Variables

Create `.env`

```
GEMINI_API_KEY=your_api_key
```

---

## 4 Run Application

```
python main.py
```

Application will start at

```
http://localhost:5000
```

---

# Assumptions

* Patient ID exists in filename
* Documents are scanned images
* OCR extraction may not be perfect
* Only key clinical information is extracted

---

# Limitations

* OCR accuracy depends on scan quality
* Entire documents are embedded as single vectors
* Extracted clinical data is not yet stored in structured format
* No PDF parsing implemented

---

# Future Improvements

If extended further, the system could include:

* Document chunking for better search accuracy
* Structured patient database
* Better UI for viewing extracted clinical highlights
* PDF support
* Source citations in answers
* Role-based authentication for clinicians
* Scalable vector database

---

# Conclusion

This prototype demonstrates how modern AI tools such as **LLMs, embeddings, and OCR** can be combined to build a **searchable medical records assistant** that improves access to clinical information from unstructured documents.
