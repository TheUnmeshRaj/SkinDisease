import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
import torch
from bs4 import BeautifulSoup
from datasets import load_dataset
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Constants
SIMILARITY_THRESHOLD = 0.5
DEFAULT_LOCALITY = "Indiranagar"
DEFAULT_LOCATION = "bangalore"
DEFAULT_BACKUP_QUERY = "dermatologist"
DEFAULT_BACKUP_MODE = "service"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"

# Disease classes
DISEASE_CLASSES = {
    0: 'Acne and Rosacea',
    1: 'Actinic Keratosis Basal Cell Carcinoma',
    2: 'Nail Fungus',
    3: 'Psoriasis Lichen Planus',
    4: 'Seborrheic Keratoses',
    5: 'Tinea Ringworm Candidiasis',
    6: 'Warts Molluscum'
}

# Device configuration
device = torch.device('cpu')
logger.info(f"Using device: {device}")

# Load datasets
try:
    logger.info("Loading datasets...")
    dataset = load_dataset("Mostafijur/Skin_disease_classify_data")
    dataset1 = load_dataset("brucewayne0459/Skin_diseases_and_care")
    logger.info("Datasets loaded successfully")
except Exception as e:
    logger.error(f"Error loading datasets: {str(e)}")
    raise

# Load models
try:
    logger.info("Loading NLP models...")
    tokenizer1 = AutoTokenizer.from_pretrained("Unmeshraj/skin-disease-detection")
    model1 = AutoModel.from_pretrained("Unmeshraj/skin-disease-detection").to(device)
    tokenizer2 = AutoTokenizer.from_pretrained("Unmeshraj/skin-disease-treatment-plan")
    model2 = AutoModel.from_pretrained("Unmeshraj/skin-disease-treatment-plan").to(device)
    
    logger.info("All models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def embed_text(text: str, tokenizer: AutoTokenizer, model: AutoModel) -> torch.Tensor:
    """Embed text using the provided tokenizer and model."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Preprocess datasets
logger.info("Preprocessing disease classification dataset...")
queries, diseases, embeddings = [], [], []
for example in dataset['train']:
    query = example['Skin_disease_classification']['query']
    disease = example['Skin_disease_classification']['disease']
    queries.append(query)
    diseases.append(disease)
    query_embedding = embed_text(query, tokenizer1, model1)
    embeddings.append(query_embedding.cpu())  # Store on CPU to save memory

logger.info("Preprocessing treatment dataset...")
topics, information, topic_embeddings = [], [], []
for example in dataset1['train']:
    topic = example['Topic']
    info = example['Information']
    topics.append(topic)
    information.append(info)
    topic_embedding = embed_text(topic, tokenizer2, model2)
    topic_embeddings.append(topic_embedding.cpu())  # Store on CPU to save memory

def find_similar_disease(input_query: str) -> str:
    """Find the most similar disease based on the input query."""
    try:
        input_embedding = embed_text(input_query, tokenizer1, model1)
        similarities = [
            cosine_similarity(input_embedding.cpu().numpy(), emb.numpy())[0][0] 
            for emb in embeddings 
        ]
        max_value = max(similarities)
        if max_value > SIMILARITY_THRESHOLD:
            return diseases[similarities.index(max_value)]
        return "Unknown"
    except Exception as e:
        logger.error(f"Error in find_similar_disease: {str(e)}")
        return "Unknown"

def find_treatment_plan(disease_name: str) -> str:
    """Find the treatment plan for the given disease."""
    try:
        if disease_name == "Unknown":
            return "No treatment plan available for the unknown disease."
        
        disease_embedding = embed_text(disease_name, tokenizer2, model2)
        similarities = [
            cosine_similarity(disease_embedding.cpu().numpy(), topic_emb.numpy())[0][0] 
            for topic_emb in topic_embeddings
        ]
        most_similar_index = similarities.index(max(similarities))
        treatment_plan = information[most_similar_index]
        
        # Format the treatment plan for better readability
        return treatment_plan.replace("*", "").replace(":", ":\n").replace(". ", ".\n")
    except Exception as e:
        logger.error(f"Error in find_treatment_plan: {str(e)}")
        return "Treatment plan could not be determined due to an error."

def fetch_doctors_from_url(url: str, headers: Dict[str, str]) -> Tuple[List[Dict[str, str]], int]:
    """Fetch doctor data from the given URL."""
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            logger.warning(f"Failed to fetch doctors: Status code {response.status_code}")
            return [], response.status_code
        
        soup = BeautifulSoup(response.text, "html.parser")
        doctor_data = []
        
        for anchor in soup.find_all("a", href=True, class_=False):
            if "/doctor/" in anchor["href"] and anchor.find("h2", class_="doctor-name"):
                name = anchor.find("h2", class_="doctor-name").get_text(strip=True)
                link = "https://www.practo.com" + anchor["href"]
                doctor_data.append({"name": name, "link": link})
        
        return doctor_data, response.status_code
    except requests.RequestException as e:
        logger.error(f"Request error in fetch_doctors_from_url: {str(e)}")
        return [], 500
    except Exception as e:
        logger.error(f"Error in fetch_doctors_from_url: {str(e)}")
        return [], 500

def get_doctor_profile(link: str, headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Get detailed profile information for a doctor."""
    try:
        profile_response = requests.get(link, headers=headers, timeout=10)
        if profile_response.status_code != 200:
            logger.warning(f"Failed to fetch doctor profile: Status code {profile_response.status_code}")
            return None
        
        profile_soup = BeautifulSoup(profile_response.text, "html.parser")
        
        # Extract qualifications
        qualifications = profile_soup.find("p", class_="c-profile__details", attrs={"data-qa-id": "doctor-qualifications"})
        qual_text = qualifications.get_text(strip=True) if qualifications else "Not available"
        
        # Extract specializations
        specializations_div = profile_soup.find("div", class_="c-profile__details", attrs={"data-qa-id": "doctor-specializations"})
        specializations = "Specializations not found"
        if specializations_div:
            spec_spans = specializations_div.find_all("span", class_="u-d-inlineblock u-spacer--right-v-thin")
            if spec_spans:
                specializations = ", ".join(span.get_text(strip=True) for span in spec_spans)
        
        # Extract experience
        experience_h2 = profile_soup.find("h2", string=lambda text: text and "Years Experience" in text)
        experience = "Experience not available"
        if experience_h2:
            experience = experience_h2.get_text(strip=True).replace("\xa0", " ")
        
        # Extract clinics
        clinics = profile_soup.find_all("p", class_="c-profile--clinic__address")
        clinic_list = [clinic.get_text(strip=True) for clinic in clinics] if clinics else ["No clinics listed"]
        
        return {
            "qualifications": qual_text,
            "specializations": specializations,
            "experience": experience,
            "clinics": clinic_list
        }
    except requests.RequestException as e:
        logger.error(f"Request error in get_doctor_profile: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error in get_doctor_profile: {str(e)}")
        return None

def fetch_doctors(
    location: str, 
    query: str, 
    mode: str, 
    backup_query: str, 
    backup_mode: str, 
    locality: str
) -> Union[List[Dict[str, Any]], Dict[str, str]]:
    """Fetch doctors based on the query and location."""
    headers = {"User-Agent": USER_AGENT}
    
    # Prepare query parameters
    if mode == "symptom":
        encoded_query = query.replace(" ", "%20")
    elif mode == "service":
        encoded_query = backup_query.replace(" ", "%20")
    else:
        encoded_query = query.replace(" ", "%20")
    
    encoded_locality = locality.replace(" ", "%20")
    
    # Build the search URL
    search_url = (
        f"https://www.practo.com/search/doctors?results_type=doctor&q="
        f"%5B%7B%22word%22%3A%22{encoded_query}%22%2C%22autocompleted%22%3Atrue%2C"
        f"%22category%22%3A%22{mode}%22%7D%2C%7B%22word%22%3A%22{encoded_locality}%22%2C"
        f"%22autocompleted%22%3Atrue%2C%22category%22%3A%22locality%22%7D%5D&city={location}"
    )
    
    # Try primary search
    doctor_data, status_code = fetch_doctors_from_url(search_url, headers)
    
    # Try backup search if primary failed
    if not doctor_data:
        logger.warning(f"Primary search failed for {encoded_query}. Trying backup query.")
        backup_url = (
            f"https://www.practo.com/search/doctors?results_type=doctor&q="
            f"%5B%7B%22word%22%3A%22{backup_query}%22%2C%22autocompleted%22%3Atrue%2C"
            f"%22category%22%3A%22{backup_mode}%22%7D%2C%7B%22word%22%3A%22{encoded_locality}%22%2C"
            f"%22autocompleted%22%3Atrue%2C%22category%22%3A%22locality%22%7D%5D&city={location}"
        )
        doctor_data, status_code = fetch_doctors_from_url(backup_url, headers)
    
    # Try final fallback
    if not doctor_data:
        logger.warning("Backup search failed. Using hardcoded fallback URL.")
        fallback_url = (
            "https://www.practo.com/search/doctors?results_type=doctor&q="
            "%5B%7B%22word%22%3A%22dermatologist%22%2C%22autocompleted%22%3Atrue%2C"
            "%22category%22%3A%22subspeciality%22%7D%2C%7B%22word%22%3A%22kengeri%22%2C"
            "%22autocompleted%22%3Atrue%2C%22category%22%3A%22locality%22%7D%5D&city=bangalore"
        )
        doctor_data, status_code = fetch_doctors_from_url(fallback_url, headers)
    
    # Return error if all attempts failed
    if not doctor_data:
        logger.error(f"All doctor search attempts failed with status code: {status_code}")
        return {"error": f"No doctors found for both primary ({query}) and backup ({backup_query}). Status Code: {status_code}"}
    
    # Process doctor profiles
    doctors_info = []
    for doctor in doctor_data:
        if doctor["name"] == "Unknown":
            continue
        
        profile_info = get_doctor_profile(doctor["link"], headers)
        if not profile_info:
            continue
        
        doctor_info = {
            "name": doctor['name'],
            "link": doctor['link'],
            **profile_info
        }
        doctors_info.append(doctor_info)
    
    # Return a sample of doctors (up to 3)
    if doctors_info:
        return random.sample(doctors_info, min(3, len(doctors_info)))
    else:
        return {"error": "Failed to retrieve detailed doctor information"}

@app.route('/api/TextAi', methods=['POST'])
def generate_result():
    """API endpoint to process text input and return disease information."""
    try:
        data = request.get_json()
        if not data or 'inputText' not in data:
            logger.warning("No input text provided in request")
            return jsonify({'error': 'No input text provided'}), 400
        
        input_query = data['inputText']
        logger.info(f"Processing query: {input_query}")
        
        # Find similar disease and treatment plan
        similar_disease = find_similar_disease(input_query)
        logger.info(f"Identified disease: {similar_disease}")
        
        treatment_plan = find_treatment_plan(similar_disease)
        
        # Only fetch doctors if the disease is known
        if similar_disease != "Unknown":
            # Get doctor recommendations
            locality = data.get('locality', DEFAULT_LOCALITY)
            location = data.get('location', DEFAULT_LOCATION)
            query = similar_disease.replace(" ", "%20")
            mode = "symptom"
            
            doctor_info = fetch_doctors(
                location, 
                query, 
                mode, 
                DEFAULT_BACKUP_QUERY, 
                DEFAULT_BACKUP_MODE, 
                locality
            )
            
            return jsonify({
                'disease': similar_disease,
                'treatment': treatment_plan,
                'doctors': doctor_info
            })
        else:
            return jsonify({
                'disease': similar_disease,
                'treatment': treatment_plan
            })
    except Exception as e:
        logger.error(f"Error in generate_result endpoint: {str(e)}")
        return jsonify({'error': f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify API is running."""
    return jsonify({'status': 'ok', 'message': 'Service is running'}), 200

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=True, port=5001)
