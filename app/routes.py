from flask import Blueprint, render_template, request, jsonify
import os

from .processing import (process_document,search_docs,ask_question,get_patient_list)

main = Blueprint("main", __name__)

UPLOAD_FOLDER = "data/uploads"


@main.route("/")
def index():
    patients = get_patient_list()
    return render_template("index.html", patients=patients)


@main.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    path = os.path.join(UPLOAD_FOLDER,file.filename)
    file.save(path)
    result = process_document(path)
    return jsonify(result)


@main.route("/search")
def search():
    query = request.args.get("q")
    patient_id = request.args.get("patient_id")
    results = search_docs(query,patient_id)
    return jsonify(results)


@main.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data["question"]
    patient_id = data["patient_id"]
    answer = ask_question(question,patient_id)
    return jsonify({"answer":answer})

