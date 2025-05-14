import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.applications import MobileNetV2, ResNet50, Xception
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess, decode_predictions as resnet_decode
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from PIL import Image
from io import BytesIO
import logging
from werkzeug.utils import secure_filename
import redis
import json
import uuid
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load the models
model_mobilenet = MobileNetV2(weights="imagenet", include_top=True)
model_resnet = ResNet50(weights="imagenet")
model_xception = Xception(weights="imagenet")

# Configure Redis for caching if available
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    redis_available = True
    logger.info("Redis connected successfully")
except:
    redis_available = False
    logger.warning("Redis not available, caching disabled")

# Create upload folder if it doesn't exist
UPLOAD_FOLDER = 'uploaded_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Category mapping for e-commerce
category_mapping = {
    # Electronics
    'laptop': 'Electronic',
    'monitor': 'Electronic',
    'desktop_computer': 'Electronic',
    'tv': 'Electronic',
    'cell_phone': 'Electronic',
    'smartphone': 'Electronic',
    'tablet': 'Electronic',
    'printer': 'Electronic',
    'camera': 'Electronic',
    'headphone': 'Electronic',
    'speaker': 'Electronic',
    
    # Clothing
    'shirt': 'clothes',
    't-shirt': 'clothes',
    'dress': 'clothes',
    'suit': 'clothes',
    'jersey': 'clothes',
    'sweater': 'clothes',
    'jacket': 'clothes',
    'hat': 'clothes',
    'cap': 'clothes',
    'shoe': 'clothes',
    'sneaker': 'clothes',
    'sandal': 'clothes',
    
    # Home & Garden
    'chair': 'Home&Garden',
    'table': 'Home&Garden',
    'sofa': 'Home&Garden',
    'bed': 'Home&Garden',
    'lamp': 'Home&Garden',
    'vase': 'Home&Garden',
    'plant': 'Home&Garden',
    'flower_pot': 'Home&Garden',
    'kitchen_appliance': 'Home&Garden',
    'refrigerator': 'Home&Garden',
    'microwave': 'Home&Garden',
    
    # Toys & Games
    'toy': 'Toys&Games',
    'teddy_bear': 'Toys&Games',
    'doll': 'Toys&Games',
    'ball': 'Toys&Games',
    'game': 'Toys&Games',
    'video_game': 'Toys&Games',
    'board_game': 'Toys&Games',
    'puzzle': 'Toys&Games',
}

# Dictionary of base prices for common items
base_pricing = {
    # Electronics - format: 'item': [min_price, avg_price, max_price]
    'laptop': [200, 600, 2000],
    'monitor': [50, 150, 500],
    'desktop_computer': [200, 500, 1500],
    'tv': [100, 300, 1000],
    'smartphone': [50, 250, 800],
    'tablet': [50, 200, 600],
    'camera': [50, 200, 800],
    'headphone': [10, 60, 300],
    'speaker': [20, 80, 400],
    
    # Clothing
    'shirt': [5, 20, 80],
    't-shirt': [5, 15, 50],
    'dress': [10, 40, 200],
    'suit': [50, 200, 800],
    'jacket': [15, 60, 300],
    'hat': [5, 15, 60],
    'shoe': [20, 60, 200],
    
    # Home & Garden
    'chair': [20, 80, 400],
    'table': [40, 150, 800],
    'sofa': [100, 400, 2000],
    'bed': [100, 300, 1200],
    'lamp': [10, 40, 200],
    'kitchen_appliance': [30, 100, 500],
    
    # Toys & Games
    'toy': [5, 20, 100],
    'teddy_bear': [5, 15, 80],
    'game': [10, 30, 100],
    'video_game': [10, 30, 60],
    'board_game': [10, 25, 80],
    'puzzle': [5, 15, 50],
}

# Price adjustment factors
condition_factors = {
    'new': 1.0,        # No adjustment for new items
    'used': 0.6,       # Used items worth about 60% of new
    'refurbished': 0.8 # Refurbished items worth about 80% of new
}

# Try to load pricing model if available
PRICE_MODEL_PATH = 'models/price_estimator.joblib'
price_model = None
price_scaler = None
price_encoder = None

try:
    if os.path.exists(PRICE_MODEL_PATH):
        price_model = joblib.load(PRICE_MODEL_PATH)
        price_scaler = joblib.load('models/price_scaler.joblib')
        price_encoder = joblib.load('models/price_encoder.joblib')
        logger.info("Price estimation model loaded successfully")
except Exception as e:
    logger.warning(f"Could not load price estimation model: {str(e)}")

def get_category_suggestion(predictions):
    """Map image recognition results to platform categories"""
    for pred in predictions:
        # Check label and description for matches
        for key in category_mapping:
            if (key in pred['label'].lower() or 
                key in pred['description'].lower()):
                return category_mapping[key]
    
    # Default fallback
    return None

def estimate_price(item_name, category, condition='used'):
    """Estimate price based on item attributes"""
    price_estimate = {
        'min_price': 0,
        'max_price': 0,
        'suggested_price': 0,
        'confidence': 'low'
    }
    
    # Try using the ML model if available
    if price_model is not None:
        try:
            # Prepare features
            features = pd.DataFrame({
                'item_name': [item_name.lower()],
                'category': [category],
                'condition': [condition]
            })
            
            # Encode categorical features
            cat_features = price_encoder.transform(features[['category', 'condition']])
            
            # Extract text features (simple keyword matching for now)
            text_features = []
            for key in base_pricing.keys():
                text_features.append(1 if key in item_name.lower() else 0)
            
            # Combine features
            all_features = np.hstack((cat_features.toarray(), np.array(text_features).reshape(1, -1)))
            
            # Scale features
            scaled_features = price_scaler.transform(all_features)
            
            # Predict price
            predicted_price = price_model.predict(scaled_features)[0]
            
            # Get prediction interval
            price_estimate['suggested_price'] = round(predicted_price, 2)
            price_estimate['min_price'] = round(predicted_price * 0.8, 2)
            price_estimate['max_price'] = round(predicted_price * 1.2, 2)
            price_estimate['confidence'] = 'high'
            
            return price_estimate
            
        except Exception as e:
            logger.warning(f"ML price estimation failed: {str(e)}")
            # Fall back to rule-based estimation
    
    # Rule-based estimation
    # Find matching item in base pricing
    base_price = None
    matched_item = None
    
    # Check for exact match
    for key in base_pricing.keys():
        if key in item_name.lower():
            matched_item = key
            base_price = base_pricing[key]
            break
    
    # If no match found and we have a category, use category averages
    if base_price is None and category:
        category_items = [base_pricing[k] for k in base_pricing.keys() 
                         if k in category_mapping and category_mapping[k] == category]
        if category_items:
            # Use average of category items
            avg_min = sum([item[0] for item in category_items]) / len(category_items)
            avg_mid = sum([item[1] for item in category_items]) / len(category_items)
            avg_max = sum([item[2] for item in category_items]) / len(category_items)
            base_price = [avg_min, avg_mid, avg_max]
    
    # If still no match, use a generic fallback
    if base_price is None:
        base_price = [20, 50, 200]  # Generic fallback prices
    
    # Apply condition factor
    condition_factor = condition_factors.get(condition, 0.7)  # Default to 0.7 if condition not recognized
    
    price_estimate['min_price'] = round(base_price[0] * condition_factor, 2)
    price_estimate['suggested_price'] = round(base_price[1] * condition_factor, 2)
    price_estimate['max_price'] = round(base_price[2] * condition_factor, 2)
    
    # Set confidence level
    if matched_item:
        price_estimate['confidence'] = 'medium'
    else:
        price_estimate['confidence'] = 'low'
    
    return price_estimate

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/recognize_image", methods=["POST"])
def recognize_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    img_file = request.files["image"]
    
    # Check if we have a cached result
    if redis_available:
        file_content = img_file.read()
        cache_key = f"img_recog_{hash(file_content)}"
        cached_result = redis_client.get(cache_key)
        if cached_result:
            logger.info("Serving cached prediction")
            return jsonify(json.loads(cached_result))
        # Reset file pointer for later processing
        img_file.seek(0)

    try:
        # Read image
        img = Image.open(BytesIO(img_file.read()))
        img = img.convert("RGB")
        img = img.resize((224, 224))
        
        # Save image for debugging/logging
        filename = secure_filename(f"{uuid.uuid4()}.jpg")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img.save(filepath)
        logger.info(f"Saved image for analysis: {filepath}")

        # Process with MobileNetV2
        x_mobilenet = image.img_to_array(img)
        x_mobilenet = np.expand_dims(x_mobilenet, axis=0)
        x_mobilenet = mobilenet_preprocess(x_mobilenet)
        
        # Process with ResNet50
        x_resnet = image.img_to_array(img)
        x_resnet = np.expand_dims(x_resnet, axis=0)
        x_resnet = resnet_preprocess(x_resnet)
        
        # Process with Xception (for better accuracy with some items)
        x_xception = image.img_to_array(img)
        x_xception = np.expand_dims(x_xception, axis=0)
        x_xception = xception_preprocess(x_xception)

        # Predict using different models
        mobilenet_preds = model_mobilenet.predict(x_mobilenet)
        resnet_preds = model_resnet.predict(x_resnet)
        xception_preds = model_xception.predict(x_xception)
        
        # Decode predictions
        mobilenet_decoded = decode_predictions(mobilenet_preds, top=3)[0]
        resnet_decoded = resnet_decode(resnet_preds, top=3)[0]
        xception_decoded = decode_predictions(xception_preds, top=3)[0]
        
        # Combine and weight predictions
        all_predictions = []
        
        # Process MobileNet predictions (weight 0.4)
        for (label, desc, prob) in mobilenet_decoded:
            all_predictions.append({
                "label": label, 
                "description": desc, 
                "probability": float(prob * 0.4),
                "model": "MobileNetV2"
            })
            
        # Process ResNet predictions (weight 0.3)
        for (label, desc, prob) in resnet_decoded:
            all_predictions.append({
                "label": label, 
                "description": desc, 
                "probability": float(prob * 0.3),
                "model": "ResNet50"
            })
            
        # Process Xception predictions (weight 0.3)
        for (label, desc, prob) in xception_decoded:
            all_predictions.append({
                "label": label, 
                "description": desc, 
                "probability": float(prob * 0.3),
                "model": "Xception"
            })
            
        # Merge duplicates and sort by probability
        merged_predictions = {}
        for pred in all_predictions:
            key = pred['description'].lower()
            if key in merged_predictions:
                merged_predictions[key]['probability'] += pred['probability']
                merged_predictions[key]['models'].append(pred['model'])
            else:
                merged_predictions[key] = {
                    'label': pred['label'],
                    'description': pred['description'],
                    'probability': pred['probability'],
                    'models': [pred['model']]
                }
                
        # Convert to list and sort
        final_predictions = list(merged_predictions.values())
        final_predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        # Limit to top 5
        final_predictions = final_predictions[:5]
        
        # For the top prediction, get category suggestion
        suggested_category = get_category_suggestion(final_predictions)
        
        # Estimate price for the top prediction
        price_estimate = None
        if final_predictions and len(final_predictions) > 0:
            top_item = final_predictions[0]['description']
            price_estimate = estimate_price(top_item, suggested_category)
        
        # Prepare response
        response = {
            "predictions": final_predictions,
            "suggested_category": suggested_category,
            "price_estimate": price_estimate
        }
        
        # Cache the result if redis is available
        if redis_available:
            redis_client.setex(
                cache_key,
                timedelta(hours=24),
                json.dumps(response)
            )

        return jsonify(response)

    except Exception as e:
        logger.exception("Error processing image")
        return jsonify({"error": str(e)}), 500

@app.route("/estimate_price", methods=["POST"])
def price_estimation_endpoint():
    """Endpoint for price estimation without image recognition"""
    data = request.json
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
        
    item_name = data.get('item_name', '')
    category = data.get('category', '')
    condition = data.get('condition', 'used')
    
    if not item_name:
        return jsonify({"error": "Item name is required"}), 400
        
    try:
        price_estimate = estimate_price(item_name, category, condition)
        return jsonify({"price_estimate": price_estimate})
    except Exception as e:
        logger.exception("Error estimating price")
        return jsonify({"error": str(e)}), 500

# Helper function for decode_predictions (customized to work with all models)
def decode_predictions(preds, top=5):
    return resnet_decode(preds, top=top)

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        "status": "healthy",
        "models": ["MobileNetV2", "ResNet50", "Xception"],
        "price_model_loaded": price_model is not None,
        "redis_available": redis_available
    })

@app.route("/stats", methods=["GET"])
def get_stats():
    """Get API usage statistics"""
    if not redis_available:
        return jsonify({"error": "Statistics not available without Redis"}), 503
    
    try:
        # Get basic stats
        total_requests = int(redis_client.get("stats:total_requests") or 0)
        successful_requests = int(redis_client.get("stats:successful_requests") or 0)
        failed_requests = int(redis_client.get("stats:failed_requests") or 0)
        
        # Get category distribution
        categories = {}
        for category in category_mapping.values():
            count = int(redis_client.get(f"stats:category:{category}") or 0)
            if count > 0:
                categories[category] = count
                
        return jsonify({
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "category_distribution": categories
        })
    except Exception as e:
        logger.exception("Error retrieving stats")
        return jsonify({"error": str(e)}), 500

# Model training endpoint - for admin use to improve price estimation
@app.route("/train_price_model", methods=["POST"])
def train_price_model():
    """Train or update the price estimation model with new data"""
    if not request.json or not 'items' in request.json:
        return jsonify({"error": "No training data provided"}), 400
        
    try:
        items = request.json['items']
        
        # Validate data format
        for item in items:
            if not all(k in item for k in ('item_name', 'category', 'condition', 'price')):
                return jsonify({"error": "Invalid item data format"}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(items)
        
        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Prepare features and target
        X = df[['item_name', 'category', 'condition']]
        y = df['price']
        
        # Encode categorical features
        encoder = OneHotEncoder(handle_unknown='ignore')
        cat_features = encoder.fit_transform(X[['category', 'condition']])
        
        # Extract text features (simple keyword matching for now)
        text_features = []
        for _, row in X.iterrows():
            item_features = []
            for key in base_pricing.keys():
                item_features.append(1 if key in row['item_name'].lower() else 0)
            text_features.append(item_features)
        
        # Combine features
        all_features = np.hstack((cat_features.toarray(), np.array(text_features)))
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(all_features)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        # Save model and preprocessing objects
        joblib.dump(model, PRICE_MODEL_PATH)
        joblib.dump(scaler, 'models/price_scaler.joblib')
        joblib.dump(encoder, 'models/price_encoder.joblib')
        
        # Update global variables
        global price_model, price_scaler, price_encoder
        price_model = model
        price_scaler = scaler
        price_encoder = encoder
        
        return jsonify({"success": "Price estimation model trained successfully"})
        
    except Exception as e:
        logger.exception("Error training price model")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)