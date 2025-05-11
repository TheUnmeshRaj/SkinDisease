import random

import requests
import torch
from bs4 import BeautifulSoup
from datasets import load_dataset
from flask import Flask, jsonify, request  # type: ignore
from flask_cors import CORS
from PIL import Image  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms
from transformers import AutoModel, AutoTokenizer

app = Flask(__name__)
CORS(app)

dataset = load_dataset("Mostafijur/Skin_disease_classify_data")
dataset1 = load_dataset("brucewayne0459/Skin_diseases_and_care")
device = torch.device('cpu')
classes = {0: 'Acne and Rosacea', 1: 'Actinic Keratosis Basal Cell Carcinoma', 
           2: 'Nail Fungus', 3: 'Psoriasis Lichen Planus', 4: 'Seborrheic Keratoses', 
           5: 'Tinea Ringworm Candidiasis', 6: 'Warts Molluscum'}

tokenizer1 = AutoTokenizer.from_pretrained("Unmeshraj/skin-disease-detection")
model1 = AutoModel.from_pretrained("Unmeshraj/skin-disease-detection")
tokenizer2 = AutoTokenizer.from_pretrained("Unmeshraj/skin-disease-treatment-plan")
model2 = AutoModel.from_pretrained("Unmeshraj/skin-disease-treatment-plan")
image_model = models.resnet18(weights=None)
image_model.fc = torch.nn.Linear(image_model.fc.in_features, len(classes))
image_model.load_state_dict(torch.load("model.pth", map_location=device))

image_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def embed_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

queries, diseases, embeddings = [], [], []
for example in dataset['train']:
    query = example['Skin_disease_classification']['query']
    disease = example['Skin_disease_classification']['disease']
    queries.append(query)
    diseases.append(disease)
    query_embedding = embed_text(query, tokenizer1, model1)
    embeddings.append(query_embedding)

topics, information, topic_embeddings = [], [], []
for example in dataset1['train']:
    topic = example['Topic']
    info = example['Information']
    topics.append(topic)
    information.append(info)
    topic_embedding = embed_text(topic, tokenizer2, model2)
    topic_embeddings.append(topic_embedding)

def find_similar_disease(input_query):
    input_embedding = embed_text(input_query, tokenizer1, model1)
    similarities = [
        cosine_similarity(input_embedding.detach().numpy(), emb.detach().numpy())[0][0] 
        for emb in embeddings
    ]
    return diseases[similarities.index(max(similarities))]

def find_treatment_plan(disease_name):
    disease_embedding = embed_text(disease_name, tokenizer2, model2)
    similarities = [
        cosine_similarity(disease_embedding.detach().numpy(), topic_emb.detach().numpy())[0][0] 
        for topic_emb in topic_embeddings
    ]
    return information[similarities.index(max(similarities))]

def predict_image(img):
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = image_model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    return classes[predicted.item()]

def fetchDoctors(location, query, mode, backupQuery, backupMode, locality):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
    }
# testting git page
    def fetch_from_url(query, mode):
        url = f"https://www.practo.com/search/doctors?results_type=doctor&q=%5B%7B%22word%22%3A%22{query}%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22{mode}%22%7D%2C%7B%22word%22%3A%22{locality}%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22locality%22%7D%5D&city={location}"
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

    def backupFetch(url):
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
    
    if not doctor_data:
        doctor_data, status_code = backupFetch("https://www.practo.com/search/doctors?results_type=doctor&q=%5B%7B%22word%22%3A%22dermatologist%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22subspeciality%22%7D%2C%7B%22word%22%3A%22kengeri%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22locality%22%7D%5D&city=bangalore")

    if not doctor_data:
        return {"error": f"No doctors found for both primary ({query}) and backup ({backupQuery}). Status Code: {status_code}"}

    doctors_info = []
    for doctor in doctor_data:
        if doctor["name"] == "Unknown":
            continue

        profile_response = requests.get(doctor["link"], headers=headers)
        if profile_response.status_code != 200:
            continue

        profile_soup = BeautifulSoup(profile_response.text, "html.parser")
        qualifications = profile_soup.find("p", class_="c-profile__details", attrs={"data-qa-id": "doctor-qualifications"})
        specializations_div = profile_soup.find("div", class_="c-profile__details", attrs={"data-qa-id": "doctor-specializations"})
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

        doctor_info = {
            "name": doctor['name'],
            "link": doctor['link'],
            "qualifications": qualifications.get_text(strip=True) if qualifications else "Not available",
            "specializations": specializations,
            "experience": experience,
            "clinics": [clinic.get_text(strip=True) for clinic in clinics] if clinics else ["No clinics listed"]
        }

        doctors_info.append(doctor_info)

    return random.sample(doctors_info, min(3, len(doctors_info)))

@app.route('/api/TextAi', methods=['POST'])
def GenResult():
    data = request.get_json()
    if 'inputText' not in data:
        return jsonify({'error': 'No input text provided'}), 400

    input_query = data['inputText']
    try:
        similar_disease = find_similar_disease(input_query)
        treatment_plan = find_treatment_plan(similar_disease).replace("*", "").replace(":", ":\n").replace(". ", ".\n")

        locality = "Indiranagar"
        location = "bangalore"
        query = similar_disease.replace(" ", "%20")
        mode = "symptom"
        backupQuery = "dermatologist"
        backupMode = "service"

        doctor_info = fetchDoctors(location, query, mode, backupQuery, backupMode, locality)
        
        return jsonify({
            'disease': similar_disease,
            'treatment': treatment_plan,
            'doctors': doctor_info
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ImageAi', methods=['POST'])
def image_ai():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')
    predicted_disease = predict_image(image)

    treatment = "Consult a dermatologist for this condition."
    return jsonify({'result': predicted_disease, 'treatment': treatment})

if __name__ == '__main__':
    app.run(debug=True,port=5001)