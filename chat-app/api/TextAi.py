import random
from typing import Any, Dict, List, Tuple

import requests
import torch
from bs4 import BeautifulSoup
from datasets import load_dataset
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms
from transformers import AutoModel, AutoTokenizer

CLASSES = {
    0: 'Acne and Rosacea',
    1: 'Actinic Keratosis Basal Cell Carcinoma',
    2: 'Nail Fungus',
    3: 'Psoriasis Lichen Planus',
    4: 'Seborrheic Keratoses',
    5: 'Tinea Ringworm Candidiasis',
    6: 'Warts Molluscum'
}
DEVICE = torch.device('cpu')
DEFAULT_LOCALITY = "Indiranagar"
DEFAULT_LOCATION = "bangalore"
DEFAULT_BACKUP_QUERY = "dermatologist"
DEFAULT_MODE = "symptom"
DEFAULT_BACKUP_MODE = "service"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/115.0 Safari/537.36"
)

app = Flask(__name__)
CORS(app)

dataset = load_dataset("Mostafijur/Skin_disease_classify_data")
dataset1 = load_dataset("brucewayne0459/Skin_diseases_and_care")

tokenizer1 = AutoTokenizer.from_pretrained("Unmeshraj/skin-disease-detection")
model1 = AutoModel.from_pretrained("Unmeshraj/skin-disease-detection")
tokenizer2 = AutoTokenizer.from_pretrained("Unmeshraj/skin-disease-treatment-plan")
model2 = AutoModel.from_pretrained("Unmeshraj/skin-disease-treatment-plan")

image_model = models.resnet18(weights=None)
image_model.fc = torch.nn.Linear(image_model.fc.in_features, len(CLASSES))
image_model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
image_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def embed_text(text: str, tokenizer: AutoTokenizer, model: AutoModel) -> torch.Tensor:
    """Generate text embeddings using the provided tokenizer and model."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def load_embeddings():
    """Load and precompute embeddings for skin disease queries and topics."""
    queries, diseases, query_embeddings = [], [], []
    for example in dataset['train']:
        query = example['Skin_disease_classification']['query']
        disease = example['Skin_disease_classification']['disease']
        queries.append(query)
        diseases.append(disease)
        query_embeddings.append(embed_text(query, tokenizer1, model1))

    topics, information, topic_embeddings = [], [], []
    for example in dataset1['train']:
        topic = example['Topic']
        info = example['Information']
        topics.append(topic)
        information.append(info)
        topic_embeddings.append(embed_text(topic, tokenizer2, model2))

    return queries, diseases, query_embeddings, topics, information, topic_embeddings

QUERIES, DISEASES, EMBEDDINGS, TOPICS, INFORMATION, TOPIC_EMBEDDINGS = load_embeddings()

def find_similar_disease(input_query: str) -> str:
    """Find the most similar disease based on input query embedding."""
    input_embedding = embed_text(input_query, tokenizer1, model1)
    similarities = [
        cosine_similarity(input_embedding.detach().numpy(), emb.detach().numpy())[0][0]
        for emb in EMBEDDINGS
    ]
    max_similarity = max(similarities)
    return DISEASES[similarities.index(max_similarity)] if max_similarity > 0.5 else "Unknown"

def find_treatment_plan(disease_name: str) -> str:
    """Retrieve treatment plan for a given disease."""
    if disease_name == "Unknown":
        return "No treatment plan available for the unknown disease."
    disease_embedding = embed_text(disease_name, tokenizer2, model2)
    similarities = [
        cosine_similarity(disease_embedding.detach().numpy(), emb.detach().numpy())[0][0]
        for emb in TOPIC_EMBEDDINGS
    ]
    return INFORMATION[similarities.index(max(similarities))]

def fetch_doctors(
    location: str,
    query: str,
    mode: str,
    backup_query: str,
    backup_mode: str,
    locality: str
) -> Dict[str, Any]:
    """Fetch doctor information based on query and location."""
    headers = {"User-Agent": USER_AGENT}

    def fetch_from_url(query: str, mode: str) -> Tuple[List[Dict[str, str]], int]:
        """Helper function to fetch doctor data from Practo."""
        query = query.replace(" ", "%20")
        url = (
            f"https://www.practo.com/search/doctors?results_type=doctor&q=%5B%7B%22word%22%3A%22{query}%22%2C"
            f"%22autocompleted%22%3Atrue%2C%22category%22%3A%22{mode}%22%7D%2C%7B%22word%22%3A%22{locality}%22%2C"
            f"%22autocompleted%22%3Atrue%2C%22category%22%3A%22locality%22%7D%5D&city={location}"
        )
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return [], response.status_code

        soup = BeautifulSoup(response.text, "html.parser")
        doctor_data = []
        for anchor in soup.find_all("a", href=True, class_=False):
            if "/doctor/" in anchor["href"] and anchor.find("h2", class_="doctor-name"):
                name = anchor.find("h2", class_="doctor-name").get_text(strip=True)
                link = "https://www.practo.com" + anchor["href"]
                doctor_data.append({"name": name, "link": link})
        return doctor_data, response.status_code

    doctor_data, status_code = fetch_from_url(query, mode)

    # Fallback to backup query if no results
    if not doctor_data:
        doctor_data, status_code = fetch_from_url(backup_query, backup_mode)

    if not doctor_data:
        return {"error": f"No doctors found. Status Code: {status_code}"}

    doctors_info = []
    for doctor in doctor_data:
        if doctor["name"] == "Unknown":
            continue

        try:
            profile_response = requests.get(doctor["link"], headers=headers, timeout=5)
            if profile_response.status_code != 200:
                continue

            profile_soup = BeautifulSoup(profile_response.text, "html.parser")
            qualifications = profile_soup.find(
                "p", class_="c-profile__details", attrs={"data-qa-id": "doctor-qualifications"}
            )
            specializations_div = profile_soup.find(
                "div", class_="c-profile__details", attrs={"data-qa-id": "doctor-specializations"}
            )
            specializations = (
                ", ".join(
                    span.get_text(strip=True)
                    for span in specializations_div.find_all("span", class_="u-d-inlineblock u-spacer--right-v-thin")
                )
                if specializations_div else "Specializations not found"
            )
            experience_h2 = profile_soup.find("h2", string=lambda text: text and "Years Experience" in text)
            experience = (
                experience_h2.get_text(strip=True).replace("\xa0", " ")
                if experience_h2 else "Experience not available"
            )
            clinics = profile_soup.find_all("p", class_="c-profile--clinic__address")

            doctors_info.append({
                "name": doctor['name'],
                "link": doctor['link'],
                "qualifications": qualifications.get_text(strip=True) if qualifications else "Not available",
                "specializations": specializations,
                "experience": experience,
                "clinics": [clinic.get_text(strip=True) for clinic in clinics] if clinics else ["No clinics listed"]
            })
        except requests.RequestException:
            continue

    return random.sample(doctors_info, min(3, len(doctors_info)))

@app.route('/api/TextAi', methods=['POST'])
def gen_result():
    """API endpoint to process text input and return disease, treatment, and doctor information."""
    data = request.get_json()
    if not data or 'inputText' not in data:
        return jsonify({'error': 'No input text provided'}), 400

    input_query = data['inputText']
    try:
        similar_disease = find_similar_disease(input_query)
        treatment_plan = find_treatment_plan(similar_disease).replace("*", "").replace(":", ":\n").replace(". ", ".\n")

        doctor_info = fetch_doctors(
            DEFAULT_LOCATION,
            similar_disease,
            DEFAULT_MODE,
            DEFAULT_BACKUP_QUERY,
            DEFAULT_BACKUP_MODE,
            DEFAULT_LOCALITY
        )

        return jsonify({
            'disease': similar_disease,
            'treatment': treatment_plan,
            'doctors': doctor_info
        })
    except Exception as e:
        return jsonify({'error': f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)