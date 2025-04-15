from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify
import cv2
import numpy as np
import os
import pickle
import requests
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import json

# Initialize Flask app
app = Flask(__name__)

# IMPORTANT: Replace with your actual Google Maps API key
GOOGLE_MAPS_API_KEY = "AIzaSyAi4eIJersq1-zyLEbfxpyNXJcYIVyPvVw"

# Create models directory if it doesn't exist
MODELS_FOLDER = 'models'
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Check if model files exist in the models folder, if not, check root directory
model_file = os.path.join(MODELS_FOLDER, 'CNN_Covid19_Xray_Version.h5')
label_encoder_file = os.path.join(MODELS_FOLDER, 'Label_encoder.pkl')

# If model files aren't in the models directory, look in the current directory
if not os.path.exists(model_file):
    model_file = 'CNN_Covid19_Xray_Version.h5'
if not os.path.exists(label_encoder_file):
    label_encoder_file = 'Label_encoder.pkl'

try:
    print(f"Loading model from: {model_file}")
    model = load_model(model_file)
    print("Model loaded successfully")

    print(f"Loading label encoder from: {label_encoder_file}")
    le = pickle.load(open(label_encoder_file, 'rb'))
    print("Label encoder loaded successfully")
except Exception as e:
    print(f"Error loading model or label encoder: {e}")
    print("Please ensure that scikit-learn is installed: pip install scikit-learn")
    import sys

    sys.exit(1)

# Path to store uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create templates directory if it doesn't exist
TEMPLATES_FOLDER = 'templates'
os.makedirs(TEMPLATES_FOLDER, exist_ok=True)


def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points using Haversine formula
    """
    from math import radians, sin, cos, sqrt, asin

    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r


def get_fallback_hospitals(latitude, longitude, disease_severity):
    """
    Provide fallback hospital data when the Google Places API fails

    Parameters:
    latitude (float): User's latitude
    longitude (float): User's longitude
    disease_severity (str): Severity level (Low, Medium, High)

    Returns:
    list: List of mock hospital details
    """
    from math import radians, sin, cos, sqrt, asin

    # Mock hospital data - we'll slightly offset these from the user's location
    mock_hospitals = [
        {
            "name": "City General Hospital - Pulmonology Department",
            "address": "123 Main St, City Center",
            "phone": "555-123-4567",
            "lat": latitude + 0.01,  # Small offset from user location
            "lon": longitude + 0.01,
            "rating": 4.5,
            "website": "https://www.example.com/city-hospital",
            "maps_url": f"https://www.google.com/maps/search/hospital/@{latitude},{longitude},14z"
        },
        {
            "name": "Community Respiratory Center",
            "address": "456 Oak Avenue",
            "phone": "555-987-6543",
            "lat": latitude - 0.008,
            "lon": longitude + 0.015,
            "rating": 4.2,
            "website": "https://www.example.com/medical-center",
            "maps_url": f"https://www.google.com/maps/search/hospital/@{latitude},{longitude},14z"
        },
        {
            "name": "Riverside Pulmonology Clinic",
            "address": "789 River Road",
            "phone": "555-456-7890",
            "lat": latitude + 0.02,
            "lon": longitude - 0.01,
            "rating": 4.7,
            "website": "https://www.example.com/riverside",
            "maps_url": f"https://www.google.com/maps/search/hospital/@{latitude},{longitude},14z"
        }
    ]

    # If high severity, add specialized hospitals
    if disease_severity == "High":
        mock_hospitals.append({
            "name": "Emergency Respiratory Care Center",
            "address": "101 Emergency Lane",
            "phone": "555-911-0000",
            "lat": latitude - 0.015,
            "lon": longitude - 0.02,
            "rating": 4.9,
            "website": "https://www.example.com/emergency-care",
            "maps_url": f"https://www.google.com/maps/search/hospital/@{latitude},{longitude},14z"
        })

    # Calculate distance for each hospital
    for hospital in mock_hospitals:
        hospital['distance'] = calculate_distance(
            latitude, longitude,
            hospital['lat'], hospital['lon']
        )

    # Sort by distance
    mock_hospitals.sort(key=lambda x: x['distance'])

    return mock_hospitals


def _try_text_search(latitude, longitude, disease_severity):
    """
    Try to find pulmonology hospitals using Text Search API
    
    This typically gives better results for specialized searches
    """
    print(f"Trying Text Search API for disease severity: {disease_severity}")
    
    # Construct query based on severity
    if disease_severity == "High":
        query = f"emergency pulmonologist hospital near {latitude},{longitude}"
    elif disease_severity == "Medium":
        query = f"pulmonology hospital respiratory care near {latitude},{longitude}"
    else:
        query = f"respiratory clinic pulmonologist near {latitude},{longitude}"
    
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": query,
        "location": f"{latitude},{longitude}",
        "radius": 15000,
        "key": GOOGLE_MAPS_API_KEY
    }
    
    try:
        print(f"Making Text Search request with query: {query}")
        response = requests.get(url, params=params)
        print(f"Text Search API response status code: {response.status_code}")
        
        data = response.json()
        print(f"Text Search API response status: {data.get('status')}")
        
        if data.get("status") == "OK" and data.get("results"):
            print(f"Found {len(data['results'])} results via Text Search API")
            hospitals = []
            
            for place in data["results"][:5]:
                # Get more details for each place
                place_id = place["place_id"]
                details_url = "https://maps.googleapis.com/maps/api/place/details/json"
                details_params = {
                    "place_id": place_id,
                    "fields": "name,formatted_address,formatted_phone_number,geometry,website,rating,url",
                    "key": GOOGLE_MAPS_API_KEY
                }

                try:
                    details_response = requests.get(details_url, params=details_params)
                    details_data = details_response.json()

                    if details_data["status"] == "OK" and "result" in details_data:
                        result = details_data["result"]

                        hospital = {
                            "name": result.get("name", ""),
                            "address": result.get("formatted_address", ""),
                            "phone": result.get("formatted_phone_number", "Not available"),
                            "rating": result.get("rating", 0),
                            "lat": result["geometry"]["location"]["lat"],
                            "lon": result["geometry"]["location"]["lng"],
                            "website": result.get("website", ""),
                            "maps_url": result.get("url", ""),
                            # Calculate distance
                            "distance": calculate_distance(
                                latitude, longitude,
                                result["geometry"]["location"]["lat"],
                                result["geometry"]["location"]["lng"]
                            )
                        }

                        hospitals.append(hospital)
                except Exception as e:
                    print(f"Error fetching details for hospital: {e}")
                    continue
            
            # Sort by distance
            hospitals.sort(key=lambda x: x['distance'])
            return hospitals
        else:
            print(f"Text Search API returned no results or error: {data.get('status')}")
            return []
            
    except Exception as e:
        print(f"Text search failed: {e}")
        return []


def _try_nearby_search(latitude, longitude, disease_severity):
    """
    Try to find hospitals using the Nearby Search API
    """
    print(f"Trying Nearby Search API for disease severity: {disease_severity}")
    
    # Define the appropriate search type based on disease severity
    if disease_severity == "High":
        keyword = "pulmonologist emergency hospital respiratory care"
        radius = 15000  # 15km radius for high severity
    elif disease_severity == "Medium":
        keyword = "pulmonology hospital lung specialist respiratory care"
        radius = 10000  # 10km radius
    else:  # Low severity
        keyword = "pulmonologist clinic respiratory care"
        radius = 5000  # 5km radius

    # Construct the URL for the Places API request
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{latitude},{longitude}",
        "radius": radius,
        "type": "hospital",
        "keyword": keyword,
        "key": GOOGLE_MAPS_API_KEY
    }

    try:
        print(f"Making Nearby Search request with keyword: {keyword}")
        response = requests.get(url, params=params)
        print(f"Nearby Search API response status code: {response.status_code}")
        
        data = response.json()
        print(f"Nearby Search API response status: {data.get('status')}")

        # Check if API returned successful results
        if data.get("status") == "OK" and len(data.get("results", [])) > 0:
            print(f"Found {len(data['results'])} results via Nearby Search API")
            # Process and return the results
            hospitals = []

            for place in data["results"][:5]:  # Limit to top 5 results
                # Get more details for each place (like phone number)
                place_id = place["place_id"]
                details_url = "https://maps.googleapis.com/maps/api/place/details/json"
                details_params = {
                    "place_id": place_id,
                    "fields": "name,formatted_address,formatted_phone_number,geometry,website,rating,url",
                    "key": GOOGLE_MAPS_API_KEY
                }

                try:
                    details_response = requests.get(details_url, params=details_params)
                    details_data = details_response.json()

                    if details_data["status"] == "OK" and "result" in details_data:
                        result = details_data["result"]

                        hospital = {
                            "name": result.get("name", ""),
                            "address": result.get("formatted_address", ""),
                            "phone": result.get("formatted_phone_number", "Not available"),
                            "rating": result.get("rating", 0),
                            "lat": result["geometry"]["location"]["lat"],
                            "lon": result["geometry"]["location"]["lng"],
                            "website": result.get("website", ""),
                            "maps_url": result.get("url", ""),
                            # Calculate distance
                            "distance": calculate_distance(
                                latitude, longitude,
                                result["geometry"]["location"]["lat"],
                                result["geometry"]["location"]["lng"]
                            )
                        }

                        hospitals.append(hospital)
                except Exception as e:
                    print(f"Error fetching details for hospital: {e}")
                    continue

            # Sort by distance
            hospitals.sort(key=lambda x: x['distance'])
            return hospitals
        else:
            print(f"Nearby Search API returned no results or error: {data.get('status')}")
            return []

    except Exception as e:
        print(f"Nearby search failed: {e}")
        return []


def _try_general_search(latitude, longitude):
    """
    Try to find any hospitals nearby without specific keywords
    
    This is the most basic search that should return results in most areas
    """
    print("Trying general hospital search as last resort")
    
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{latitude},{longitude}",
        "radius": 15000,
        "type": "hospital",
        "key": GOOGLE_MAPS_API_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get("status") == "OK" and data.get("results"):
            print(f"Found {len(data['results'])} results via general hospital search")
            hospitals = []
            
            for place in data["results"][:5]:
                # Get more details for each place
                place_id = place["place_id"]
                details_url = "https://maps.googleapis.com/maps/api/place/details/json"
                details_params = {
                    "place_id": place_id,
                    "fields": "name,formatted_address,formatted_phone_number,geometry,website,rating,url",
                    "key": GOOGLE_MAPS_API_KEY
                }

                try:
                    details_response = requests.get(details_url, params=details_params)
                    details_data = details_response.json()

                    if details_data["status"] == "OK" and "result" in details_data:
                        result = details_data["result"]

                        hospital = {
                            "name": result.get("name", ""),
                            "address": result.get("formatted_address", ""),
                            "phone": result.get("formatted_phone_number", "Not available"),
                            "rating": result.get("rating", 0),
                            "lat": result["geometry"]["location"]["lat"],
                            "lon": result["geometry"]["location"]["lng"],
                            "website": result.get("website", ""),
                            "maps_url": result.get("url", ""),
                            # Calculate distance
                            "distance": calculate_distance(
                                latitude, longitude,
                                result["geometry"]["location"]["lat"],
                                result["geometry"]["location"]["lng"]
                            )
                        }

                        hospitals.append(hospital)
                except Exception as e:
                    print(f"Error fetching details for hospital: {e}")
                    continue
            
            # Sort by distance
            hospitals.sort(key=lambda x: x['distance'])
            return hospitals
        else:
            print(f"General search API returned no results or error: {data.get('status')}")
            return []
            
    except Exception as e:
        print(f"General search failed: {e}")
        return []


def get_nearby_hospitals(latitude, longitude, disease_severity):
    """
    Find nearby hospitals using Google Places API with multi-tiered approach
    
    Parameters:
    latitude (float): User's latitude
    longitude (float): User's longitude
    disease_severity (str): Severity level (Low, Medium, High)
    
    Returns:
    list: List of hospital details
    """
    print(f"\n--- Starting hospital search for severity: {disease_severity} ---")
    print(f"User location: {latitude}, {longitude}")
    
    # Try each search method in order of specialization
    
    # 1. Try Text Search API first (best for specialized searches)
    hospitals = _try_text_search(latitude, longitude, disease_severity)
    if hospitals:
        print(f"Successfully found {len(hospitals)} hospitals using Text Search API")
        return hospitals
    
    # 2. Try Nearby Search API if text search fails
    hospitals = _try_nearby_search(latitude, longitude, disease_severity)
    if hospitals:
        print(f"Successfully found {len(hospitals)} hospitals using Nearby Search API")
        return hospitals
    
    # 3. Try general hospital search without keywords
    hospitals = _try_general_search(latitude, longitude)
    if hospitals:
        print(f"Successfully found {len(hospitals)} hospitals using general search")
        return hospitals
    
    # 4. Use fallback mock data as last resort
    print("All API searches failed, using fallback mock data")
    return get_fallback_hospitals(latitude, longitude, disease_severity)


def determine_severity(disease_type, confidence_score):
    """
    Determine the severity level based on disease type and confidence score
    """
    severity = "Low"

    if disease_type == "Normal":
        severity = "Low"
    elif disease_type == "COVID-19":
        if confidence_score > 0.8:
            severity = "High"
        elif confidence_score > 0.6:
            severity = "Medium"
        else:
            severity = "Low"
    elif disease_type == "Viral Pneumonia":
        if confidence_score > 0.9:
            severity = "High"
        elif confidence_score > 0.7:
            severity = "Medium"
        else:
            severity = "Low"

    return severity


def process_image(image_path):
    """Process an X-ray image and return the diagnosis prediction"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found at path: {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (150, 150))
        image_normalized = image_resized / 255.0
        image_input = np.expand_dims(image_normalized, axis=0)
        predictions = model.predict(image_input)
        predicted_index = np.argmax(predictions)
        confidence_score = predictions[0][predicted_index]
        predicted_label = le.inverse_transform([predicted_index])[0]
        return predicted_label, confidence_score
    except Exception as e:
        print(f"Error processing image: {e}")
        return "Error", 0.0


@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html', google_maps_api_key=GOOGLE_MAPS_API_KEY)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads and process images"""
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Get user location if provided
    user_lat = request.form.get('latitude', '37.7749')  # Default to San Francisco coordinates
    user_lon = request.form.get('longitude', '-122.4194')

    try:
        user_lat = float(user_lat)
        user_lon = float(user_lon)
    except ValueError:
        user_lat = 37.7749
        user_lon = -122.4194

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        try:
            # Process the X-ray image
            predicted_label, confidence_score = process_image(file_path)

            # Determine severity
            severity = determine_severity(predicted_label, confidence_score)

            # Get hospital recommendations based on location and severity
            hospitals = get_nearby_hospitals(user_lat, user_lon, severity)

            return render_template('result.html',
                                   filename=filename,
                                   predicted_label=predicted_label,
                                   confidence_score=confidence_score,
                                   hospitals=hospitals,
                                   severity=severity,
                                   google_maps_api_key=GOOGLE_MAPS_API_KEY)
        except Exception as e:
            return f"Error processing image: {e}"


@app.route('/camera')
def camera():
    """Render the camera page"""
    return render_template('camera.html', google_maps_api_key=GOOGLE_MAPS_API_KEY)


@app.route('/debug_api')
def debug_api():
    """
    Debug endpoint to test Google Places API directly
    
    This route allows direct testing of the API to diagnose issues
    """
    lat = request.args.get('lat', '37.7749')
    lon = request.args.get('lon', '-122.4194')
    keyword = request.args.get('keyword', 'pulmonology hospital')
    search_type = request.args.get('type', 'nearby')  # 'nearby', 'text', or 'general'
    
    if search_type == 'text':
        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        params = {
            "query": f"{keyword} near {lat},{lon}",
            "key": GOOGLE_MAPS_API_KEY
        }
    elif search_type == 'general':
        url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            "location": f"{lat},{lon}",
            "radius": 15000,
            "type": "hospital",
            "key": GOOGLE_MAPS_API_KEY
        }
    else:  # nearby
        url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            "location": f"{lat},{lon}",
            "radius": 15000,
            "type": "hospital",
            "keyword": keyword,
            "key": GOOGLE_MAPS_API_KEY
        }
    
    try:
        response = requests.get(url, params=params)
        return jsonify({
            "request_url": url,
            "request_params": params,
            "status_code": response.status_code,
            "api_response": response.json()
        })
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    # Check if templates exist, if not, create them
    index_template = os.path.join(TEMPLATES_FOLDER, 'index.html')
    result_template = os.path.join(TEMPLATES_FOLDER, 'result.html')
    camera_template = os.path.join(TEMPLATES_FOLDER, 'camera.html')

    if not os.path.exists(index_template) or not os.path.exists(result_template) or not os.path.exists(camera_template):
        print("Creating template files...")
        # Create template files from the documents provided
        # You should manually copy your HTML files to the templates directory
        print("Please ensure your HTML template files are in the 'templates' directory")

    app.run(debug=True)