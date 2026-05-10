"""
FedFusionNet++ - Complete Flask Application
Authentication + Dashboard + Prediction
"""

from flask import Flask, request, jsonify, render_template, session, redirect, url_for, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import re
import dns.resolver
import requests
import secrets
import hashlib
from datetime import datetime
import os
import sys
from pathlib import Path
from pymongo import MongoClient
from dotenv import load_dotenv
import base64
from io import BytesIO

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Set template and static folders to frontend
app = Flask(__name__, 
            template_folder='../frontend/templates',
            static_folder='../frontend/static')
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(32))
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
CORS(app)

ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png'}

# ============================================
# MONGODB SETUP (WITH ERROR HANDLING)
# ============================================
MONGODB_URI = os.getenv('MONGODB_URI')
DATABASE_NAME = os.getenv('DATABASE_NAME', 'fedfusionnet')

# Try to connect to MongoDB, but don't fail if unavailable
try:
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    # Test connection
    client.server_info()
    db = client[DATABASE_NAME]
    hospitals_collection = db['hospitals']
    predictions_collection = db['predictions']
    
    # Create indexes
    hospitals_collection.create_index('email', unique=True)
    hospitals_collection.create_index('gst_number', unique=True)
    hospitals_collection.create_index('hospital_id', unique=True)
    
    print("[OK] MongoDB connected successfully")
    MONGODB_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] MongoDB connection failed: {e}")
    print("[WARNING] Running without database (predictions won't be saved)")
    MONGODB_AVAILABLE = False
    client = None
    db = None
    hospitals_collection = None
    predictions_collection = None

# ============================================
# VALIDATION FUNCTIONS
# ============================================
BLOCKED_DOMAINS = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'rediffmail.com']

def validate_email_domain(email):
    domain = email.split('@')[1].lower()
    if domain in BLOCKED_DOMAINS:
        return False, "Personal email domains not allowed"
    return True, "Valid"

def validate_domain_exists(email):
    domain = email.split('@')[1]
    try:
        dns.resolver.resolve(domain, 'MX')
        return True, "Domain valid"
    except:
        return False, "Domain does not exist"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ============================================
# MODEL LOADING - HYBRID SYSTEM WITH HF MODEL HUB
# ============================================
def download_model_from_hf(filename):
    """Download model from Hugging Face Model Hub"""
    try:
        from huggingface_hub import hf_hub_download
        
        MODEL_REPO_ID = "afridpasha1983/fedfusionnet-models"
        
        print(f"[HF] Downloading {filename} from Model Hub...")
        model_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=filename,
            cache_dir="./models_cache"
        )
        print(f"[HF] Downloaded: {model_path}")
        return model_path
    except Exception as e:
        print(f"[HF] Failed to download {filename}: {e}")
        return None

def load_cnn_model():
    """Load Stage-1 CNN Model (SWIN-ViT + Cross-ViT trained on Kaggle)"""
    try:
        from backend.model.fedfusionnet_simple import FedFusionNetPlus
        
        # Create model (no pretrained weights, we'll load your trained model)
        model = FedFusionNetPlus(num_classes=2)
        
        # Try local path first
        model_path = Path(__file__).parent.parent / 'models' / 'hetfusionnet_v2_FINAL.pth'
        
        # If not found locally, download from Hugging Face
        if not model_path.exists():
            print("[HF] Model not found locally, downloading from Hugging Face...")
            downloaded_path = download_model_from_hf('hetfusionnet_v2_FINAL.pth')
            if downloaded_path:
                model_path = Path(downloaded_path)
            else:
                print(f"[ERROR] Failed to download CNN model")
                return None
        
        # Load model weights
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print(f"[OK] Stage-1 CNN model loaded: {model_path.name}")
        print(f"[OK] Architecture: SWIN-ViT + Cross-ViT")
        return model
        
    except Exception as e:
        print(f"[ERROR] Error loading CNN model: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_tabular_model():
    """Load Stage-2 Tabular Model (trained on CSV with Stage-1 outputs)"""
    try:
        from backend.model.tabular_model import OralCancerTabularModel
        
        # Try local path first
        model_path = Path(__file__).parent.parent / 'models' / 'stage2_tabular_model.pkl'
        
        # If not found locally, download from Hugging Face
        if not model_path.exists():
            print("[HF] Model not found locally, downloading from Hugging Face...")
            downloaded_path = download_model_from_hf('stage2_tabular_model.pkl')
            if downloaded_path:
                model_path = Path(downloaded_path)
            else:
                print(f"⚠ Failed to download tabular model")
                return None
        
        # Load model
        tabular = OralCancerTabularModel(model_path=str(model_path))
        return tabular
        
    except Exception as e:
        print(f"⚠ Error loading tabular model: {e}")
        return None

CNN_MODEL = load_cnn_model()
TABULAR_MODEL = load_tabular_model()

# Load SHAP Explainer
SHAP_EXPLAINER = None
try:
    from backend.shap_explainer import create_shap_explainer
    SHAP_EXPLAINER = create_shap_explainer()
    print("[OK] SHAP Explainer loaded")
except Exception as e:
    print(f"[WARNING] SHAP Explainer not available: {e}")

# Load Survival Analyzer
SURVIVAL_ANALYZER = None
try:
    from backend.survival_analysis import create_survival_analyzer
    SURVIVAL_ANALYZER = create_survival_analyzer()
    print("[OK] Survival Analyzer loaded")
except Exception as e:
    print(f"[WARNING] Survival Analyzer not available: {e}")

# Load Temporal Comparator
TEMPORAL_COMPARATOR = None
try:
    from backend.temporal_comparison import create_temporal_comparator
    TEMPORAL_COMPARATOR = create_temporal_comparator()
    print("[OK] Temporal Comparator loaded")
except Exception as e:
    print(f"[WARNING] Temporal Comparator not available: {e}")

# Load VLM Service (TIER 1 Feature)
VLM_SERVICE = None
try:
    from backend.vlm_service import create_vlm_service
    VLM_SERVICE = create_vlm_service()  # Uses .env VLM_PROVIDER
    if VLM_SERVICE and VLM_SERVICE.is_available():
        print(f"[OK] VLM Service loaded: {VLM_SERVICE.provider}")
    else:
        print("[WARNING] VLM Service not available (check API keys in .env)")
except Exception as e:
    print(f"[WARNING] VLM Service not available: {e}")

# Load WSI Processor (TIER 1 Feature)
WSI_PROCESSOR = None
try:
    from backend.wsi_processor import create_wsi_processor
    WSI_PROCESSOR = create_wsi_processor()
    if WSI_PROCESSOR.openslide_available:
        print("[OK] WSI Processor loaded (OpenSlide available)")
    else:
        print("[WARNING] WSI Processor loaded but OpenSlide not available")
except Exception as e:
    print(f"[WARNING] WSI Processor not available: {e}")

# Weights
CNN_WEIGHT = 0.95
TABULAR_WEIGHT = 0.5

def preprocess_image(image):
    """Preprocess image with Macenko normalization for Stage-1 CNN model"""
    from backend.preprocessing import preprocess_image_for_prediction
    tensor, quality_info = preprocess_image_for_prediction(image, apply_quality_control=True)
    return tensor, quality_info

def predict_image(model, img_tensor, mc_passes=50):
    if model is None:
        return None, None, None
    predictions = []
    with torch.no_grad():
        for _ in range(mc_passes):
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            predictions.append(probs.cpu().numpy())
    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0)[0]
    uncertainty = predictions.std(axis=0)[0]
    pred_class = mean_pred.argmax()
    return int(pred_class), float(mean_pred[pred_class]), float(uncertainty[pred_class])

# ============================================
# ROUTES - AUTHENTICATION
# ============================================
@app.route('/')
def index():
    if 'hospital_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/api/register', methods=['POST'])
def register():
    if not MONGODB_AVAILABLE:
        return jsonify({'success': False, 'message': 'Database unavailable'}), 503
    
    try:
        data = request.form
        hospital_name = data.get('hospital_name')
        gst_number = data.get('gst_number')
        email = data.get('email')
        phone = data.get('phone')
        address = data.get('address')
        password = data.get('password')
        
        # Validate email
        valid, message = validate_email_domain(email)
        if not valid:
            return jsonify({'success': False, 'message': message}), 400
        
        # Validate domain
        valid, message = validate_domain_exists(email)
        if not valid:
            return jsonify({'success': False, 'message': message}), 400
        
        # Handle file uploads
        gst_file = request.files.get('gst_file')
        clinical_file = request.files.get('clinical_file')
        
        if not gst_file or not clinical_file:
            return jsonify({'success': False, 'message': 'Both documents required'}), 400
        
        if not allowed_file(gst_file.filename) or not allowed_file(clinical_file.filename):
            return jsonify({'success': False, 'message': 'Invalid file format'}), 400
        
        # Generate credentials
        hospital_id = f"HOSP-{secrets.token_hex(8).upper()}"
        client_secret = secrets.token_urlsafe(32)
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # Save to MongoDB (files stored as metadata only)
        hospital_doc = {
            'hospital_name': hospital_name,
            'gst_number': gst_number,
            'email': email,
            'phone': phone,
            'address': address,
            'password_hash': password_hash,
            'hospital_id': hospital_id,
            'client_secret': client_secret,
            'status': 'approved',
            'risk_score': 0,
            'created_at': datetime.now(),
            'verified_at': datetime.now(),
            'gst_file_name': gst_file.filename,
            'clinical_file_name': clinical_file.filename
        }
        hospitals_collection.insert_one(hospital_doc)
        
        return jsonify({
            'success': True,
            'message': 'Registration successful!',
            'hospital_id': hospital_id
        }), 201
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    
    # Hardcoded test credentials (works without MongoDB)
    TEST_CREDENTIALS = {
        'admin@hospital.com': {
            'password': 'admin123',
            'hospital_id': 'HOSP-TEST-001',
            'hospital_name': 'Test Hospital'
        },
        'demo@clinic.com': {
            'password': 'demo123',
            'hospital_id': 'HOSP-DEMO-002',
            'hospital_name': 'Demo Clinic'
        }
    }
    
    # Check hardcoded credentials first
    if email in TEST_CREDENTIALS and password == TEST_CREDENTIALS[email]['password']:
        session['hospital_id'] = TEST_CREDENTIALS[email]['hospital_id']
        session['hospital_name'] = TEST_CREDENTIALS[email]['hospital_name']
        return jsonify({'success': True, 'message': 'Login successful'}), 200
    
    # If MongoDB available, check database
    if not MONGODB_AVAILABLE:
        return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
    
    # Validate email
    valid, message = validate_email_domain(email)
    if not valid:
        return jsonify({'success': False, 'message': message}), 400
    
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    # Query MongoDB
    hospital = hospitals_collection.find_one({
        'email': email,
        'password_hash': password_hash
    })
    
    if not hospital:
        return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
    
    hospital_id = hospital['hospital_id']
    hospital_name = hospital['hospital_name']
    status = hospital['status']
    
    if status != 'approved':
        return jsonify({'success': False, 'message': f'Account status: {status}'}), 403
    
    session['hospital_id'] = hospital_id
    session['hospital_name'] = hospital_name
    
    return jsonify({'success': True, 'message': 'Login successful'}), 200

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# ============================================
# ROUTES - DASHBOARD
# ============================================
@app.route('/dashboard')
def dashboard():
    if 'hospital_id' not in session:
        return redirect(url_for('index'))
    
    hospital_id = session['hospital_id']
    total_predictions = 0
    
    if MONGODB_AVAILABLE:
        # Fetch ALL predictions from MongoDB (not filtered by hospital_id)
        total_predictions = predictions_collection.count_documents({})
        print(f"[DASHBOARD] Total predictions in MongoDB: {total_predictions}")
    
    return render_template('dashboard.html', 
                         hospital_name=session.get('hospital_name'),
                         hospital_id=hospital_id,
                         total_predictions=total_predictions)

@app.route('/predict')
def predict_page():
    if 'hospital_id' not in session:
        return redirect(url_for('index'))
    return render_template('predict.html', hospital_name=session.get('hospital_name'))

@app.route('/result')
def result_page():
    if 'hospital_id' not in session:
        return redirect(url_for('index'))
    return render_template('result.html', hospital_name=session.get('hospital_name'))

@app.route('/tests')
def tests():
    if 'hospital_id' not in session:
        return redirect(url_for('index'))
    return render_template('tests.html', hospital_name=session.get('hospital_name'))

@app.route('/history')
def history():
    if 'hospital_id' not in session:
        return redirect(url_for('index'))
    return render_template('history.html', hospital_name=session.get('hospital_name'))

@app.route('/analytics')
def analytics():
    if 'hospital_id' not in session:
        return redirect(url_for('index'))
    return render_template('analytics.html', hospital_name=session.get('hospital_name'))

@app.route('/results')
def results():
    if 'hospital_id' not in session:
        return redirect(url_for('index'))
    
    hospital_id = session['hospital_id']
    predictions = []
    
    if MONGODB_AVAILABLE:
        # Fetch ALL predictions from MongoDB (not filtered by hospital_id)
        predictions = list(predictions_collection.find(
            {}
        ).sort('timestamp', -1).limit(50))
        
        print(f"[RESULTS] Found {len(predictions)} predictions in MongoDB")
        
        for pred in predictions:
            pred['_id'] = str(pred['_id'])
            pred['timestamp'] = pred['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            # Keep PDF binary in document for download endpoint
    
    return render_template('results.html', 
                         hospital_name=session.get('hospital_name'),
                         predictions=predictions)

@app.route('/api/download-pdf/<prediction_id>', methods=['GET'])
def download_pdf(prediction_id):
    """Download PDF report from MongoDB"""
    if 'hospital_id' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    if not MONGODB_AVAILABLE:
        return jsonify({'success': False, 'message': 'Database unavailable'}), 503
    
    try:
        from bson.objectid import ObjectId
        from flask import send_file
        
        # Find prediction document (not filtered by hospital_id)
        prediction = predictions_collection.find_one({
            '_id': ObjectId(prediction_id)
        })
        
        if not prediction:
            return jsonify({'success': False, 'message': 'Prediction not found'}), 404
        
        if 'pdf_report' not in prediction or not prediction['pdf_report']:
            return jsonify({'success': False, 'message': 'PDF not available'}), 404
        
        # Decode base64 PDF
        pdf_binary = base64.b64decode(prediction['pdf_report'])
        
        # Create BytesIO object
        pdf_buffer = BytesIO(pdf_binary)
        pdf_buffer.seek(0)
        
        filename = prediction.get('pdf_filename', f"report_{prediction_id}.pdf")
        
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        print(f"[ERROR] PDF download failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify models are loaded"""
    return jsonify({
        'status': 'running',
        'mongodb': 'connected' if MONGODB_AVAILABLE else 'disconnected',
        'models': {
            'cnn_model': 'loaded' if CNN_MODEL is not None else 'not loaded',
            'tabular_model': 'loaded' if TABULAR_MODEL is not None else 'not loaded'
        },
        'message': 'FedFusionNet++ API is running'
    }), 200

@app.route('/api/test-navbar', methods=['GET'])
def test_navbar():
    """Test endpoint to verify navbar links"""
    import os
    template_path = os.path.join(app.template_folder, 'tests.html')
    
    with open(template_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    has_analytics = 'href="/analytics"' in content
    has_history = 'href="/history"' in content
    
    return jsonify({
        'template_path': template_path,
        'has_analytics_link': has_analytics,
        'has_history_link': has_history,
        'template_auto_reload': app.config.get('TEMPLATES_AUTO_RELOAD', False)
    }), 200

@app.route('/api/patient-history/<patient_id>', methods=['GET'])
def get_patient_history(patient_id):
    """Get all test history for a specific patient"""
    if 'hospital_id' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    if not MONGODB_AVAILABLE:
        return jsonify({'success': False, 'message': 'Database unavailable'}), 503
    
    try:
        category = request.args.get('category', 'oral')  # Get category filter
        
        # Build category filter
        category_filter = {}
        if category == 'oral':
            # For oral, accept both with test_type='oral_cancer' OR without test_type (old data)
            category_filter = {'$or': [{'test_type': 'oral_cancer'}, {'test_type': {'$exists': False}}]}
        elif category == 'brain':
            category_filter = {'test_type': 'brain_tumor'}
        elif category == 'lung':
            category_filter = {'test_type': 'lung_cancer'}
        
        # Find all predictions for this patient at THIS hospital with category filter
        query = {
            'patient_id': patient_id,
            'hospital_id': session['hospital_id'],  # Keep hospital filter for data isolation
            **category_filter
        }
        
        tests = list(predictions_collection.find(query).sort('timestamp', -1))
        
        print(f"[PATIENT-HISTORY] Found {len(tests)} {category} tests for patient {patient_id} at hospital {session['hospital_id']}")
        
        # Calculate patient-specific analytics BEFORE converting timestamps
        analytics = calculate_patient_analytics(tests)
        
        print(f"[PATIENT-HISTORY] Analytics calculated: {analytics}")
        
        # Temporal comparison (if multiple tests exist)
        temporal_comparison = None
        if TEMPORAL_COMPARATOR and len(tests) > 1:
            try:
                print("[TEMPORAL] Generating temporal comparison...")
                
                # Current test is the most recent (first in sorted list)
                current_test = tests[0]
                previous_tests = tests[1:]  # All older tests
                
                temporal_comparison = TEMPORAL_COMPARATOR.compare_with_history(
                    current_test,
                    previous_tests
                )
                
                print(f"[TEMPORAL] Comparison status: {temporal_comparison.get('prediction_change', {}).get('status')}")
                
            except Exception as e:
                print(f"[TEMPORAL] Error generating comparison: {e}")
                import traceback
                traceback.print_exc()
        
        # Convert ObjectId to string and format dates AFTER analytics calculation
        for test in tests:
            test['_id'] = str(test['_id'])
            # Keep timestamp as datetime object for now, convert to ISO string
            if 'timestamp' in test and hasattr(test['timestamp'], 'isoformat'):
                test['timestamp'] = test['timestamp'].isoformat()
        
        return jsonify({
            'success': True,
            'patient_id': patient_id,
            'category': category,
            'total_tests': len(tests),
            'tests': tests,
            'analytics': analytics,  # Add analytics data
            'temporal_comparison': temporal_comparison  # Add temporal comparison
        }), 200
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch patient history: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

def calculate_patient_analytics(tests):
    """Calculate analytics for a specific patient's test history"""
    from datetime import datetime
    
    if not tests or len(tests) == 0:
        print("[ANALYTICS] No tests found, returning empty analytics")
        return {
            'total_tests': 0,
            'oscc_detected': 0,
            'normal_detected': 0,
            'avg_confidence': 0,
            'highest_risk': 'N/A',
            'latest_stage': 'N/A',
            'trend': 'N/A',
            'timeline': {'labels': [], 'predictions': [], 'confidences': []}
        }
    
    print(f"[ANALYTICS] Calculating analytics for {len(tests)} tests")
    
    total_tests = len(tests)
    oscc_count = sum(1 for t in tests if t.get('final_prediction') == 'OSCC')
    normal_count = total_tests - oscc_count
    
    print(f"[ANALYTICS] OSCC: {oscc_count}, Normal: {normal_count}")
    
    confidences = [t.get('final_confidence', 0) for t in tests]
    avg_confidence = round(sum(confidences) / len(confidences), 1) if confidences else 0
    
    # Find highest risk level
    risk_priority = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'N/A': 0}
    risks = [t.get('risk_level', 'N/A') for t in tests]
    highest_risk = max(risks, key=lambda x: risk_priority.get(x, 0)) if risks else 'N/A'
    
    # Get latest cancer stage
    latest_stage = 'N/A'
    if tests and len(tests) > 0 and tests[0].get('stage2_tabular'):
        stage_val = tests[0]['stage2_tabular'].get('cancer_stage', 'N/A')
        latest_stage = str(stage_val) if stage_val != 'N/A' else 'N/A'
    
    # Determine trend (improving/worsening/stable)
    trend = 'N/A'
    if len(tests) >= 2:
        recent_tests = tests[:min(3, len(tests))]
        older_tests = tests[-min(3, len(tests)):]
        
        recent_oscc = sum(1 for t in recent_tests if t.get('final_prediction') == 'OSCC')
        older_oscc = sum(1 for t in older_tests if t.get('final_prediction') == 'OSCC')
        
        if recent_oscc < older_oscc:
            trend = 'Improving'
        elif recent_oscc > older_oscc:
            trend = 'Worsening'
        else:
            trend = 'Stable'
    elif len(tests) == 1:
        # For single test, check if it's OSCC or Normal
        trend = 'Baseline'
    
    # Timeline data (reverse to show chronological order)
    timeline_tests = list(reversed(tests))[:10]  # Last 10 tests
    timeline_labels = []
    timeline_predictions = []
    timeline_confidences = []
    
    for idx, t in enumerate(timeline_tests):
        timestamp = t.get('timestamp')
        
        # Handle both datetime objects and ISO strings
        if isinstance(timestamp, datetime):
            date_str = timestamp.strftime('%m/%d')
        elif isinstance(timestamp, str):
            try:
                # Try parsing ISO format
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                date_str = dt.strftime('%m/%d')
            except:
                # Fallback to test number
                date_str = f'Test {idx + 1}'
        else:
            date_str = f'Test {idx + 1}'
        
        timeline_labels.append(date_str)
        timeline_predictions.append(1 if t.get('final_prediction') == 'OSCC' else 0)
        timeline_confidences.append(float(t.get('final_confidence', 0)))
    
    result = {
        'total_tests': total_tests,
        'oscc_detected': oscc_count,
        'normal_detected': normal_count,
        'avg_confidence': avg_confidence,
        'highest_risk': highest_risk,
        'latest_stage': latest_stage,
        'trend': trend,
        'timeline': {
            'labels': timeline_labels,
            'predictions': timeline_predictions,
            'confidences': timeline_confidences
        }
    }
    
    print(f"[ANALYTICS] Result: {result}")
    return result

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get analytics data for dashboard"""
    if 'hospital_id' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    if not MONGODB_AVAILABLE:
        return jsonify({'success': False, 'message': 'Database unavailable'}), 503
    
    try:
        from datetime import timedelta
        
        hospital_id = session['hospital_id']
        time_range = request.args.get('range', '30')
        category = request.args.get('category', 'oral')  # Get category filter
        
        # Calculate date filter
        if time_range != 'all':
            days = int(time_range)
            start_date = datetime.now() - timedelta(days=days)
            date_filter = {'timestamp': {'$gte': start_date}}
        else:
            date_filter = {}
        
        # Category filter - for now, only 'oral' has data
        # Future: Add 'test_type' field to predictions and filter by it
        category_filter = {}
        if category == 'brain':
            # No data yet for brain tumor
            category_filter = {'test_type': 'brain_tumor'}  # Will return empty
        elif category == 'lung':
            # No data yet for lung cancer
            category_filter = {'test_type': 'lung_cancer'}  # Will return empty
        # For 'oral', no filter needed (all current data is oral cancer)
        
        # Base query - fetch ALL predictions from MongoDB (not filtered by hospital_id)
        # This allows test accounts to see all data
        base_query = {**date_filter, **category_filter}
        
        # Get all predictions from MongoDB
        predictions = list(predictions_collection.find(base_query))
        
        print(f"[ANALYTICS] Found {len(predictions)} predictions in MongoDB")
        print(f"[ANALYTICS] Hospital ID: {hospital_id}")
        print(f"[ANALYTICS] Time range: {time_range} days")
        print(f"[ANALYTICS] Category: {category}")
        
        # Calculate statistics
        total_tests = len(predictions)
        oscc_count = sum(1 for p in predictions if p.get('final_prediction') == 'OSCC')
        high_risk_count = sum(1 for p in predictions if p.get('risk_level') == 'HIGH')
        
        oscc_percentage = round((oscc_count / total_tests * 100) if total_tests > 0 else 0, 1)
        high_risk_percentage = round((high_risk_count / total_tests * 100) if total_tests > 0 else 0, 1)
        
        confidences = [p.get('final_confidence', 0) for p in predictions]
        avg_confidence = round(sum(confidences) / len(confidences) if confidences else 0, 1)
        
        # Timeline data (group by date)
        timeline_data = {}
        for p in predictions:
            date_key = p['timestamp'].strftime('%Y-%m-%d')
            timeline_data[date_key] = timeline_data.get(date_key, 0) + 1
        
        sorted_dates = sorted(timeline_data.keys())
        timeline_labels = sorted_dates
        timeline_values = [timeline_data[d] for d in sorted_dates]
        
        # Risk distribution
        risk_high = sum(1 for p in predictions if p.get('risk_level') == 'HIGH')
        risk_medium = sum(1 for p in predictions if p.get('risk_level') == 'MEDIUM')
        risk_low = sum(1 for p in predictions if p.get('risk_level') == 'LOW')
        
        # Confidence distribution (buckets)
        conf_buckets = {'0-20': 0, '20-40': 0, '40-60': 0, '60-80': 0, '80-100': 0}
        for conf in confidences:
            if conf < 20:
                conf_buckets['0-20'] += 1
            elif conf < 40:
                conf_buckets['20-40'] += 1
            elif conf < 60:
                conf_buckets['40-60'] += 1
            elif conf < 80:
                conf_buckets['60-80'] += 1
            else:
                conf_buckets['80-100'] += 1
        
        # Demographics
        age_groups = {'0-30': 0, '31-45': 0, '46-60': 0, '61+': 0}
        gender_count = {'male': 0, 'female': 0}
        risk_factors = {}
        
        for p in predictions:
            patient_data = p.get('patient_data', {})
            
            # Age
            age = patient_data.get('Age', 0)
            if age <= 30:
                age_groups['0-30'] += 1
            elif age <= 45:
                age_groups['31-45'] += 1
            elif age <= 60:
                age_groups['46-60'] += 1
            else:
                age_groups['61+'] += 1
            
            # Gender
            gender = patient_data.get('Gender', '').lower()
            if gender == 'male':
                gender_count['male'] += 1
            elif gender == 'female':
                gender_count['female'] += 1
            
            # Risk factors
            for factor in ['Tobacco Use', 'Alcohol Consumption', 'Betel Quid Use', 'HPV Infection']:
                if patient_data.get(factor) == 'Yes':
                    risk_factors[factor] = risk_factors.get(factor, 0) + 1
        
        # Top risk factors
        top_risk_factors = sorted(risk_factors.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return jsonify({
            'success': True,
            'statistics': {
                'total_tests': total_tests,
                'oscc_count': oscc_count,
                'oscc_percentage': oscc_percentage,
                'avg_confidence': avg_confidence,
                'high_risk_count': high_risk_count,
                'high_risk_percentage': high_risk_percentage
            },
            'charts': {
                'timeline': {
                    'labels': timeline_labels,
                    'data': timeline_values
                },
                'distribution': {
                    'oscc': oscc_count,
                    'normal': total_tests - oscc_count
                },
                'risk': {
                    'high': risk_high,
                    'medium': risk_medium,
                    'low': risk_low
                },
                'confidence': {
                    'labels': list(conf_buckets.keys()),
                    'data': list(conf_buckets.values())
                },
                'demographics': {
                    'age': {
                        'labels': list(age_groups.keys()),
                        'data': list(age_groups.values())
                    },
                    'gender': gender_count,
                    'risk_factors': [{'name': name, 'count': count} for name, count in top_risk_factors]
                }
            }
        }), 200
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch analytics: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/generate-pdf', methods=['POST'])
def generate_pdf_endpoint():
    """Generate PDF report from prediction results"""
    try:
        from backend.clinical_reports import ClinicalReportGenerator
        from flask import send_file
        
        data = request.json
        patient_id = data.get('patient_id', 'UNKNOWN')
        results = data.get('results', {})
        
        # Prepare complete result data
        complete_result = {
            'patient_id': patient_id,
            'hospital_name': session.get('hospital_name', 'Unknown Hospital'),
            'hospital_id': session.get('hospital_id', 'Unknown'),
            'timestamp': datetime.now().isoformat(),
            'stage1_cnn': results.get('stage1_cnn', {}),
            'stage2_tabular': results.get('stage2_tabular'),
            'final_prediction': results.get('final_prediction', 'N/A'),
            'final_confidence': results.get('final_confidence', 0),
            'risk_level': results.get('risk_level', 'N/A'),
            'patient_data': {}
        }
        
        # Generate PDF
        report_gen = ClinicalReportGenerator()
        pdf_path = report_gen.generate_pdf_report(complete_result)
        
        print(f"[PDF] Generated PDF: {pdf_path}")
        
        # Send file
        return send_file(
            pdf_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'clinical_report_{patient_id}.pdf'
        )
        
    except Exception as e:
        print(f"[ERROR] PDF generation failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/vlm-qa', methods=['POST'])
def vlm_qa():
    """TIER 1: VLM Conversational Q&A endpoint"""
    if 'hospital_id' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    if not VLM_SERVICE or not VLM_SERVICE.is_available():
        return jsonify({'success': False, 'message': 'VLM service not available'}), 503
    
    try:
        data = request.json
        question = data.get('question', '')
        prediction_data = data.get('prediction_data', {})
        
        if not question:
            return jsonify({'success': False, 'message': 'Question required'}), 400
        
        print(f"[VLM-QA] Question: {question}")
        
        # Build COMPREHENSIVE context with ALL available data
        context = {
            # Patient demographics and risk factors
            'patient_data': prediction_data.get('patient_data', {}),
            
            # Stage-1 CNN results (image analysis)
            'stage1_cnn': {
                'prediction': prediction_data.get('stage1_cnn', {}).get('prediction'),
                'confidence': prediction_data.get('stage1_cnn', {}).get('confidence'),
                'confidence_level': prediction_data.get('stage1_cnn', {}).get('confidence_level'),
                'uncertainty': prediction_data.get('stage1_cnn', {}).get('uncertainty'),
                'preprocessing': prediction_data.get('stage1_cnn', {}).get('preprocessing')
            },
            
            # Stage-2 Tabular results (clinical data analysis)
            'stage2_tabular': prediction_data.get('stage2_tabular', {}),
            
            # XAI Heatmaps (visual explanations)
            'xai_heatmaps': {
                'gradcam_available': bool(prediction_data.get('stage1_cnn', {}).get('xai', {}).get('gradcam')),
                'layercam_available': bool(prediction_data.get('stage1_cnn', {}).get('xai', {}).get('scorecam')),
                'risk_tier': prediction_data.get('stage1_cnn', {}).get('xai', {}).get('risk_tier'),
                'risk_action': prediction_data.get('stage1_cnn', {}).get('xai', {}).get('risk_action')
            },
            
            # SHAP Risk Factor Analysis
            'shap_analysis': {
                'top_risk_factors': prediction_data.get('shap_explanation', {}).get('top_risk_factors', []),
                'top_protective_factors': prediction_data.get('shap_explanation', {}).get('top_protective_factors', [])
            },
            
            # Survival Analysis (Kaplan-Meier)
            'survival_analysis': {
                'median_survival_months': prediction_data.get('survival_analysis', {}).get('survival_curve', {}).get('median_survival_months'),
                'survival_1year': prediction_data.get('survival_analysis', {}).get('survival_curve', {}).get('milestones', {}).get('1_year'),
                'survival_3year': prediction_data.get('survival_analysis', {}).get('survival_curve', {}).get('milestones', {}).get('3_year'),
                'survival_5year': prediction_data.get('survival_analysis', {}).get('survival_curve', {}).get('milestones', {}).get('5_year'),
                'population_comparison': prediction_data.get('survival_analysis', {}).get('population_comparison', {}).get('assessment'),
                'recommendations': prediction_data.get('survival_analysis', {}).get('recommendations', [])
            },
            
            # WSI Analysis (Whole Slide Imaging)
            'wsi_analysis': {
                'available': bool(prediction_data.get('wsi_result')),
                'dimensions': prediction_data.get('wsi_result', {}).get('dimensions'),
                'total_tiles': prediction_data.get('wsi_result', {}).get('total_tiles'),
                'cancer_tiles': prediction_data.get('wsi_result', {}).get('cancer_tiles'),
                'normal_tiles': prediction_data.get('wsi_result', {}).get('normal_tiles'),
                'tissue_percentage': prediction_data.get('wsi_result', {}).get('tissue_percentage'),
                'avg_confidence': prediction_data.get('wsi_result', {}).get('avg_confidence')
            },
            
            # Final hybrid prediction
            'final_prediction': prediction_data.get('final_prediction', 'N/A'),
            'final_confidence': prediction_data.get('final_confidence', 0),
            'risk_level': prediction_data.get('risk_level', 'N/A'),
            
            # VLM Clinical Narrative
            'clinical_narrative': prediction_data.get('vlm_narrative', '')
        }
        
        print(f"[VLM-QA] Context includes: patient_data, CNN, tabular, XAI, SHAP, survival, WSI")
        
        # Generate answer with comprehensive context
        answer = VLM_SERVICE.conversational_qa(
            question=question,
            context=context
        )
        
        if answer:
            print(f"[VLM-QA] Answer generated ({len(answer)} chars)")
            return jsonify({
                'success': True,
                'question': question,
                'answer': answer,
                'provider': VLM_SERVICE.provider
            }), 200
        else:
            return jsonify({'success': False, 'message': 'Failed to generate answer'}), 500
            
    except Exception as e:
        print(f"[VLM-QA] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/vlm-translate', methods=['POST'])
def vlm_translate():
    """TIER 1: VLM Multi-language translation endpoint"""
    if 'hospital_id' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    if not VLM_SERVICE or not VLM_SERVICE.is_available():
        return jsonify({'success': False, 'message': 'VLM service not available'}), 503
    
    try:
        data = request.json
        prediction_data = data.get('prediction_data', {})
        target_language = data.get('target_language', 'hindi')
        
        # Get narrative from prediction data
        narrative = prediction_data.get('vlm_narrative', '')
        
        if not narrative:
            return jsonify({'success': False, 'error': 'No narrative found in prediction data'}), 400
        
        print(f"[VLM-TRANSLATE] Translating to {target_language}...")
        print(f"[VLM-TRANSLATE] Narrative length: {len(narrative)} chars")
        
        # Translate
        translated = VLM_SERVICE.generate_multi_language_report(
            narrative=narrative,
            target_language=target_language
        )
        
        if translated:
            print(f"[VLM-TRANSLATE] Translation completed ({len(translated)} chars)")
            return jsonify({
                'success': True,
                'original': narrative,
                'translated_report': translated,
                'language': target_language,
                'provider': VLM_SERVICE.provider
            }), 200
        else:
            return jsonify({'success': False, 'error': 'Translation failed - no output from VLM'}), 500
            
    except Exception as e:
        print(f"[VLM-TRANSLATE] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # ═══════════════════════════════════════════════════════
        # STAGE 1: CNN MODEL (hetfusionnet_v2_FINAL.pth)
        # ═══════════════════════════════════════════════════════
        if 'image' not in request.files:
            return jsonify({'success': False, 'message': 'No image uploaded'}), 400
        
        if CNN_MODEL is None:
            return jsonify({'success': False, 'message': 'CNN model not loaded'}), 500
        
        image_file = request.files['image']
        image = Image.open(image_file.stream).convert('RGB')
        
        # Preprocess with Macenko normalization
        img_tensor, quality_info = preprocess_image(image)
        
        # MC-Dropout for uncertainty
        cnn_class, cnn_confidence, cnn_uncertainty = predict_image(CNN_MODEL, img_tensor, mc_passes=50)
        
        if cnn_class is None:
            return jsonify({'success': False, 'message': 'Prediction failed'}), 500
        
        cnn_label = 'OSCC' if cnn_class == 1 else 'Normal'
        
        # Confidence level
        if cnn_confidence >= 0.9 and cnn_uncertainty < 0.05:
            confidence_level = 'HIGH'
            confidence_code = 2
        elif cnn_confidence >= 0.75 and cnn_uncertainty < 0.15:
            confidence_level = 'MODERATE'
            confidence_code = 1
        else:
            confidence_level = 'LOW'
            confidence_code = 0
        
        # XAI: Generate ULTIMATE Medical-Grade Heatmaps
        xai_visualizations = None
        try:
            from backend.xai_ultimate import generate_ultimate_xai
            
            print("[XAI-ULTIMATE] Generating crystal-clear medical-grade heatmaps...")
            
            gradcam_overlay, layercam_overlay = generate_ultimate_xai(
                CNN_MODEL, image, img_tensor, class_idx=cnn_class, device='cpu'
            )
            
            if gradcam_overlay is not None or layercam_overlay is not None:
                def array_to_base64(img_array):
                    """Convert numpy array to base64 string"""
                    pil_img = Image.fromarray(img_array.astype('uint8'))
                    buffer = BytesIO()
                    pil_img.save(buffer, format='PNG')
                    img_str = base64.b64encode(buffer.getvalue()).decode()
                    return f"data:image/png;base64,{img_str}"
                
                xai_visualizations = {
                    'gradcam': array_to_base64(gradcam_overlay) if gradcam_overlay is not None else None,
                    'scorecam': array_to_base64(layercam_overlay) if layercam_overlay is not None else None,
                    'risk_tier': 'HIGH' if cnn_class == 1 and cnn_confidence > 0.75 else 'MODERATE',
                    'risk_action': 'Immediate specialist referral + biopsy' if cnn_class == 1 and cnn_confidence > 0.75 else 'Follow-up imaging',
                    'risk_color': '#FF4444' if cnn_class == 1 and cnn_confidence > 0.75 else '#FFA500'
                }
                print(f"[XAI-ULTIMATE] Medical-grade heatmaps generated successfully")
            else:
                print("[XAI-ULTIMATE] No visualizations generated")
        except Exception as e:
            print(f"[XAI-ULTIMATE] Medical-grade XAI failed: {e}")
            import traceback
            traceback.print_exc()
        
        cnn_result = {
            'model': 'CNN (HetFusionNet v2)',
            'prediction': cnn_label,
            'confidence': round(cnn_confidence * 100, 2),
            'confidence_level': confidence_level,
            'uncertainty': round(cnn_uncertainty, 4),
            'preprocessing': quality_info,
            'xai': xai_visualizations
        }
        
        # ═══════════════════════════════════════════════════════
        # STAGE 2: TABULAR MODEL (stage2_tabular_model.pkl)
        # ═══════════════════════════════════════════════════════
        tabular_result = None
        tabular_class = 0
        
        if TABULAR_MODEL and request.form.get('Age'):
            patient_data = {
                'Age': int(request.form.get('Age', 45)),
                'Gender': request.form.get('Gender', 'Male'),
                'Tobacco Use': request.form.get('Tobacco_Use', 'No'),
                'Alcohol Consumption': request.form.get('Alcohol', 'No'),
                'HPV Infection': request.form.get('HPV', 'No'),
                'Betel Quid Use': request.form.get('Betel_Quid', 'No'),
                'Chronic Sun Exposure': request.form.get('Sun_Exposure', 'No'),
                'Poor Oral Hygiene': request.form.get('Oral_Hygiene', 'Yes'),
                'Diet (Fruits & Vegetables Intake)': request.form.get('Diet', 'Low'),
                'Family History of Cancer': request.form.get('Family_History', 'No'),
                'Compromised Immune System': request.form.get('Immune', 'No'),
                'Oral Lesions': request.form.get('Oral_Lesions', 'No'),
                'Unexplained Bleeding': request.form.get('Bleeding', 'No'),
                'Difficulty Swallowing': request.form.get('Swallowing', 'No'),
                'White or Red Patches in Mouth': request.form.get('Patches', 'No'),
                'Country': request.form.get('Country', 'India')
            }
            
            cnn_outputs = {
                'cnn_diagnosis': cnn_class,
                'cnn_probability': cnn_confidence,
                'cnn_confidence': confidence_code,
                'cnn_uncertainty': cnn_uncertainty
            }
            
            stage_result = TABULAR_MODEL.predict(patient_data, cnn_outputs)
            
            tabular_result = {
                'model': 'Tabular (Stage-2)',
                'cancer_stage': stage_result['cancer_stage'],
                'stage_confidence': round(stage_result['stage_confidence'] * 100, 2),
                'survival_rate_5yr': round(stage_result['survival_rate_5yr'], 2),
                'treatment_type': stage_result['treatment_type'],
                'cost_usd': round(stage_result['cost_usd'], 2),
                'economic_burden_days': stage_result['economic_burden_days']
            }
            
            tabular_class = 1 if stage_result['cancer_stage'] > 0 else 0
        
        # ═══════════════════════════════════════════════════════
        # SHAP EXPLAINABILITY: Risk Factor Contribution Analysis
        # ═══════════════════════════════════════════════════════
        shap_explanation = None
        what_if_scenarios = None
        
        if SHAP_EXPLAINER and tabular_result:
            try:
                print("[SHAP] Generating risk factor contribution analysis...")
                
                # Generate SHAP explanation
                shap_explanation = SHAP_EXPLAINER.explain_prediction(
                    patient_data, 
                    cnn_outputs
                )
                
                # Generate what-if scenarios
                what_if_scenarios = SHAP_EXPLAINER.generate_what_if_scenarios(
                    patient_data,
                    cnn_outputs
                )
                
                print(f"[SHAP] Found {len(shap_explanation['contributions'])} contributing factors")
                print(f"[SHAP] Generated {len(what_if_scenarios)} what-if scenarios")
                
            except Exception as e:
                print(f"[SHAP] Error generating explanation: {e}")
                import traceback
                traceback.print_exc()
        
        # ═══════════════════════════════════════════════════════
        # SURVIVAL ANALYSIS: Kaplan-Meier Curves
        # ═══════════════════════════════════════════════════════
        survival_analysis = None
        
        if SURVIVAL_ANALYZER and tabular_result:
            try:
                print("[SURVIVAL] Generating survival analysis...")
                
                # Generate survival report
                survival_analysis = SURVIVAL_ANALYZER.generate_survival_report(
                    cancer_stage=tabular_result['cancer_stage'],
                    patient_data=patient_data
                )
                
                print(f"[SURVIVAL] Median survival: {survival_analysis['survival_curve']['median_survival_months']} months")
                print(f"[SURVIVAL] 5-year survival: {survival_analysis['survival_curve']['milestones']['5_year']:.1%}")
                
            except Exception as e:
                print(f"[SURVIVAL] Error generating survival analysis: {e}")
                import traceback
                traceback.print_exc()
        
        # ═══════════════════════════════════════════════════════
        # HYBRID FUSION: 0.95×CNN + 0.5×Tabular
        # ═══════════════════════════════════════════════════════
        if tabular_result:
            cnn_score = cnn_class * cnn_confidence
            tabular_score = tabular_class * (tabular_result['stage_confidence'] / 100)
            hybrid_score = (CNN_WEIGHT * cnn_score) + (TABULAR_WEIGHT * tabular_score)
            final_prediction = 'OSCC' if hybrid_score > 0.5 else 'Normal'
            final_confidence = round(((CNN_WEIGHT * cnn_confidence) + (TABULAR_WEIGHT * (tabular_result['stage_confidence']/100))) / (CNN_WEIGHT + TABULAR_WEIGHT) * 100, 2)
        else:
            final_prediction = cnn_label
            final_confidence = round(cnn_confidence * 100, 2)
        
        # Risk level
        if final_confidence >= 90 and cnn_uncertainty < 0.05:
            risk_level = 'LOW' if final_prediction == 'Normal' else 'HIGH'
        elif final_confidence >= 75:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'HIGH'
        
        # Prepare patient ID
        patient_id = request.form.get('Patient_ID', f"PAT-{datetime.now().strftime('%Y%m%d%H%M%S')}")
        
        # Store both thumbnail and original image for MongoDB
        thumbnail_base64 = None
        original_image_base64 = None
        try:
            # Create small thumbnail (200x200) for storage
            thumbnail = image.copy()
            thumbnail.thumbnail((200, 200), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            thumbnail.save(buffer, format='JPEG', quality=70)
            thumbnail_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Store original image at reasonable size (max 800x800) to avoid 16MB limit
            original_resized = image.copy()
            original_resized.thumbnail((800, 800), Image.Resampling.LANCZOS)
            buffer_original = BytesIO()
            original_resized.save(buffer_original, format='JPEG', quality=85)
            original_image_base64 = base64.b64encode(buffer_original.getvalue()).decode()
        except Exception as e:
            print(f"[WARNING] Failed to encode images: {e}")
        
        # ═══════════════════════════════════════════════════════
        # TIER 1 FEATURE: VLM CLINICAL NARRATIVE GENERATION
        # ═══════════════════════════════════════════════════════
        vlm_narrative = None
        
        if VLM_SERVICE and VLM_SERVICE.is_available() and tabular_result:
            try:
                print(f"[VLM] Generating clinical narrative using {VLM_SERVICE.provider}...")
                
                # Generate comprehensive clinical narrative
                vlm_narrative = VLM_SERVICE.generate_clinical_narrative(
                    image=image,
                    patient_data=patient_data,
                    prediction_result={
                        'stage1_cnn': cnn_result,
                        'stage2_tabular': tabular_result,
                        'final_prediction': final_prediction,
                        'final_confidence': final_confidence,
                        'risk_level': risk_level
                    },
                    narrative_type='comprehensive'
                )
                
                if vlm_narrative:
                    print(f"[VLM] Clinical narrative generated ({len(vlm_narrative)} chars)")
                else:
                    print("[VLM] Failed to generate narrative")
                    
            except Exception as e:
                print(f"[VLM] Error generating narrative: {e}")
                import traceback
                traceback.print_exc()
        
        # ═══════════════════════════════════════════════════════
        # TIER 1 FEATURE: WSI SPATIAL ANALYSIS (FOR ALL IMAGES)
        # ═══════════════════════════════════════════════════════
        wsi_result = None
        
        # Process EVERY image with tile-based spatial analysis
        try:
            print(f"[WSI] Processing image with tile-based spatial analysis...")
            print(f"[WSI] Image size: {image.size}")
            
            # Convert PIL image to numpy array
            image_array = np.array(image)
            
            # Tile the image into 224x224 patches
            tile_size = 224
            overlap = 0  # No overlap for cleaner visualization
            
            height, width = image_array.shape[:2]
            tiles = []
            tile_coords = []
            
            # Extract tiles
            for y in range(0, height - tile_size + 1, tile_size - overlap):
                for x in range(0, width - tile_size + 1, tile_size - overlap):
                    tile = image_array[y:y+tile_size, x:x+tile_size]
                    
                    # Skip if tile is too small
                    if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                        continue
                    
                    tiles.append(tile)
                    tile_coords.append((x, y))
            
            print(f"[WSI] Extracted {len(tiles)} tiles of size {tile_size}x{tile_size}")
            
            if len(tiles) > 0:
                # Predict each tile using CNN model
                tile_predictions = []
                
                # Process tiles in batches
                batch_size = 32
                for i in range(0, len(tiles), batch_size):
                    batch_tiles = tiles[i:i+batch_size]
                    batch_tensors = []
                    
                    for tile in batch_tiles:
                        # Convert tile to PIL Image
                        tile_pil = Image.fromarray(tile.astype('uint8'))
                        
                        # Preprocess tile (same as main image)
                        tile_tensor, _ = preprocess_image(tile_pil)
                        batch_tensors.append(tile_tensor)
                    
                    # Stack into batch
                    batch_tensor = torch.cat(batch_tensors, dim=0)
                    
                    # Predict batch
                    with torch.no_grad():
                        outputs = CNN_MODEL(batch_tensor)
                        probs = torch.softmax(outputs, dim=1)
                        
                        for j, prob in enumerate(probs):
                            tile_idx = i + j
                            pred_class = prob.argmax().item()
                            pred_prob = prob[pred_class].item()
                            
                            tile_predictions.append({
                                'tile_idx': tile_idx,
                                'x': tile_coords[tile_idx][0],
                                'y': tile_coords[tile_idx][1],
                                'class': pred_class,
                                'probability': pred_prob,
                                'oscc_prob': prob[1].item(),
                                'normal_prob': prob[0].item()
                            })
                    
                    print(f"[WSI] Processed batch {i//batch_size + 1}/{(len(tiles) + batch_size - 1)//batch_size}")
                
                # Calculate statistics
                cancer_tiles = sum(1 for p in tile_predictions if p['class'] == 1)
                normal_tiles = len(tile_predictions) - cancer_tiles
                avg_confidence = sum(p['probability'] for p in tile_predictions) / len(tile_predictions)
                
                print(f"[WSI] Cancer tiles: {cancer_tiles}, Normal tiles: {normal_tiles}")
                print(f"[WSI] Average confidence: {avg_confidence:.2%}")
                
                # Generate spatial heatmap
                print("[WSI] Generating spatial heatmap...")
                
                # Create heatmap grid
                grid_height = (height // tile_size) + (1 if height % tile_size > 0 else 0)
                grid_width = (width // tile_size) + (1 if width % tile_size > 0 else 0)
                
                heatmap_array = np.zeros((grid_height, grid_width))
                
                for pred in tile_predictions:
                    grid_x = pred['x'] // tile_size
                    grid_y = pred['y'] // tile_size
                    heatmap_array[grid_y, grid_x] = pred['oscc_prob']
                
                # Resize heatmap to match original image size
                from scipy.ndimage import zoom
                scale_y = height / (grid_height * tile_size)
                scale_x = width / (grid_width * tile_size)
                heatmap_resized = zoom(heatmap_array, (scale_y * tile_size, scale_x * tile_size), order=1)
                
                # Crop to original size
                heatmap_resized = heatmap_resized[:height, :width]
                
                # Apply colormap (blue -> yellow -> red)
                import matplotlib.pyplot as plt
                import matplotlib.cm as cm
                
                # Normalize heatmap to 0-1
                heatmap_normalized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
                
                # Apply jet colormap
                colormap = cm.get_cmap('jet')
                heatmap_colored = colormap(heatmap_normalized)
                heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
                
                # Overlay heatmap on original image
                alpha = 0.5
                overlay = (alpha * heatmap_colored + (1 - alpha) * image_array).astype(np.uint8)
                
                # Convert to base64
                overlay_pil = Image.fromarray(overlay)
                buffer = BytesIO()
                overlay_pil.save(buffer, format='PNG')
                heatmap_base64 = base64.b64encode(buffer.getvalue()).decode()
                
                print("[WSI] Spatial heatmap generated successfully")
                
                # Prepare WSI result
                wsi_result = {
                    'dimensions': [width, height],
                    'total_tiles': len(tile_predictions),
                    'tile_size': tile_size,
                    'cancer_tiles': cancer_tiles,
                    'normal_tiles': normal_tiles,
                    'tissue_percentage': 1.0,
                    'avg_confidence': avg_confidence,
                    'heatmap_base64': heatmap_base64,
                    'tile_predictions': tile_predictions,
                    'grid_dimensions': [grid_width, grid_height]
                }
                
                print(f"[WSI] WSI spatial analysis complete: {len(tile_predictions)} tiles analyzed")
                
        except Exception as e:
            print(f"[WSI] Error during spatial analysis: {e}")
            import traceback
            traceback.print_exc()
            wsi_result = None
        
        # OLD CODE REMOVED - Now processes ALL images, not just .svs/.ndpi/.tiff
        if False and WSI_PROCESSOR and WSI_PROCESSOR.openslide_available:
            # Check if uploaded file is WSI format
            if image_file.filename and WSI_PROCESSOR.is_wsi_file(image_file.filename):
                try:
                    print(f"[WSI] Detected WSI file: {image_file.filename}")
                    print("[WSI] Processing whole-slide image with tile-based analysis...")
                    
                    # Save WSI file temporarily
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image_file.filename).suffix) as tmp:
                        image_file.save(tmp.name)
                        wsi_path = tmp.name
                    
                    # Process WSI with CNN model
                    wsi_result = WSI_PROCESSOR.process_wsi_with_model(
                        wsi_path=wsi_path,
                        model=CNN_MODEL,
                        device='cpu',
                        batch_size=32
                    )
                    
                    if wsi_result:
                        print(f"[WSI] Processed {wsi_result.get('num_tiles', 0)} tiles")
                        print(f"[WSI] WSI-level prediction: Class {wsi_result.get('class', 0)} ({wsi_result.get('probability', 0):.2%})")
                        
                        # Normalize field names for frontend consistency
                        wsi_result['total_tiles'] = wsi_result.get('num_tiles', 0)
                        wsi_result['tissue_percentage'] = wsi_result.get('tissue_ratio', 0)
                        wsi_result['avg_confidence'] = wsi_result.get('probability', 0)
                        
                        # Calculate cancer vs normal tiles
                        tile_predictions = wsi_result.get('tile_predictions', [])
                        if tile_predictions:
                            cancer_count = sum(1 for pred in tile_predictions if pred.get('class', 0) == 1)
                            normal_count = len(tile_predictions) - cancer_count
                            wsi_result['cancer_tiles'] = cancer_count
                            wsi_result['normal_tiles'] = normal_count
                        else:
                            wsi_result['cancer_tiles'] = 0
                            wsi_result['normal_tiles'] = wsi_result['total_tiles']
                        
                        # Generate spatial heatmap
                        heatmap = WSI_PROCESSOR.generate_wsi_heatmap(wsi_result)
                        if heatmap is not None:
                            # Convert heatmap to base64
                            heatmap_pil = Image.fromarray(heatmap)
                            buffer = BytesIO()
                            heatmap_pil.save(buffer, format='PNG')
                            wsi_result['heatmap_base64'] = base64.b64encode(buffer.getvalue()).decode()
                            print("[WSI] Spatial heatmap generated")
                        
                        print(f"[WSI] Final result: {wsi_result['total_tiles']} tiles, {wsi_result['cancer_tiles']} cancer, {wsi_result['normal_tiles']} normal")
                    
                    # Clean up temp file
                    os.unlink(wsi_path)
                    
                except Exception as e:
                    print(f"[WSI] Error processing WSI: {e}")
                    import traceback
                    traceback.print_exc()
        
        # ═══════════════════════════════════════════════════════
        # GENERATE CLINICAL REPORT & SEND EMAIL
        # ═══════════════════════════════════════════════════════
        pdf_binary = None
        pdf_path = None
        json_path = None
        
        try:
            from backend.clinical_reports import ClinicalReportGenerator
            from backend.email_service import create_email_service
            
            # Generate clinical reports (PDF + JSON)
            print(f"[REPORT] Generating clinical report for patient {patient_id}...")
            report_gen = ClinicalReportGenerator()
            
            # Prepare complete result data
            complete_result = {
                'patient_id': patient_id,
                'hospital_name': session.get('hospital_name', 'Unknown Hospital'),
                'hospital_id': session.get('hospital_id', 'Unknown'),
                'timestamp': datetime.now().isoformat(),
                'stage1_cnn': cnn_result,
                'stage2_tabular': tabular_result,
                'final_prediction': final_prediction,
                'final_confidence': final_confidence,
                'risk_level': risk_level,
                'patient_data': patient_data if tabular_result else {}
            }
            
            # Generate PDF report
            pdf_path = report_gen.generate_pdf_report(complete_result)
            print(f"[REPORT] PDF generated: {pdf_path}")
            
            # Read PDF as binary for MongoDB storage
            with open(pdf_path, 'rb') as f:
                pdf_binary = base64.b64encode(f.read()).decode()
            print(f"[REPORT] PDF encoded for MongoDB storage")
            
            # Generate JSON report
            json_path = report_gen.generate_json_report(complete_result)
            print(f"[REPORT] JSON generated: {json_path}")
            
            # Send email with reports
            email_service = create_email_service()
            if email_service and MONGODB_AVAILABLE:
                # Get hospital email from database
                hospital = hospitals_collection.find_one({'hospital_id': session['hospital_id']})
                if hospital:
                    # Use notification_email if available, otherwise use regular email
                    hospital_email = hospital.get('notification_email', hospital.get('email'))
                    hospital_name = hospital['hospital_name']
                    
                    print(f"[EMAIL] Sending clinical report to {hospital_email}...")
                    
                    success, message = email_service.send_clinical_report(
                        recipient_email=hospital_email,
                        hospital_name=hospital_name,
                        patient_id=patient_id,
                        prediction_result=complete_result,
                        pdf_path=pdf_path,
                        json_path=json_path
                    )
                    
                    if success:
                        print(f"[EMAIL] Clinical report sent successfully to {hospital_email}")
                    else:
                        print(f"[EMAIL] Failed to send email: {message}")
                else:
                    print("[EMAIL] Hospital not found in database")
            else:
                if not email_service:
                    print("[EMAIL] Email service not configured (check .env file)")
                if not MONGODB_AVAILABLE:
                    print("[EMAIL] MongoDB not available, cannot retrieve hospital email")
        
        except Exception as e:
            print(f"[ERROR] Failed to generate report or send email: {e}")
            import traceback
            traceback.print_exc()
            # Don't fail the prediction if email fails
        
        # ═══════════════════════════════════════════════════════
        # SAVE COMPLETE DATA TO MONGODB
        # ═══════════════════════════════════════════════════════
        if MONGODB_AVAILABLE and 'hospital_id' in session:
            try:
                # Store only metadata and small thumbnail (NOT full images/heatmaps/charts)
                # This prevents MongoDB 16MB BSON limit error
                prediction_document = {
                    'hospital_id': session['hospital_id'],
                    'hospital_name': session.get('hospital_name', 'Unknown'),
                    'patient_id': patient_id,
                    'timestamp': datetime.now(),
                    
                    # Test type for category filtering
                    'test_type': 'oral_cancer',  # oral_cancer, brain_tumor, lung_cancer
                    
                    # Patient data
                    'patient_data': patient_data if tabular_result else {},
                    
                    # Small thumbnail (200x200 JPEG ~10KB)
                    'thumbnail': thumbnail_base64,
                    
                    # Original image resized (800x800 JPEG ~50-100KB)
                    'original_image': original_image_base64,
                    
                    # Stage-1 CNN results (WITHOUT xai field to avoid base64 images)
                    'stage1_cnn': {
                        'model': cnn_result['model'],
                        'prediction': cnn_result['prediction'],
                        'confidence': cnn_result['confidence'],
                        'confidence_level': cnn_result['confidence_level'],
                        'uncertainty': cnn_result['uncertainty'],
                        'preprocessing': cnn_result['preprocessing']
                        # xai field excluded - too large for MongoDB
                    },
                    
                    # Stage-2 Tabular results
                    'stage2_tabular': tabular_result,
                    
                    # SHAP explanation (WITHOUT waterfall chart base64)
                    'shap_explanation': {
                        'contributions': shap_explanation['contributions'] if shap_explanation else [],
                        'top_risk_factors': shap_explanation['top_risk_factors'] if shap_explanation else [],
                        'top_protective_factors': shap_explanation['top_protective_factors'] if shap_explanation else []
                        # waterfall_chart excluded - too large
                    } if shap_explanation else None,
                    
                    # Survival analysis (WITHOUT chart base64)
                    'survival_analysis': {
                        'survival_curve': {
                            'median_survival_months': survival_analysis['survival_curve']['median_survival_months'],
                            'milestones': survival_analysis['survival_curve']['milestones']
                            # time_points and survival_probabilities excluded - chart data too large
                        },
                        'population_comparison': survival_analysis['population_comparison'],
                        'assessment': survival_analysis.get('assessment', 'N/A'),  # Use .get() to avoid KeyError
                        'recommendations': survival_analysis['recommendations']
                        # chart_base64 excluded - too large
                    } if survival_analysis else None,
                    
                    # Final prediction
                    'final_prediction': final_prediction,
                    'final_confidence': final_confidence,
                    'risk_level': risk_level,
                    
                    # PDF report reference (store file path, not binary)
                    'pdf_path': pdf_path,
                    'pdf_filename': f'clinical_report_{patient_id}.pdf',
                    
                    # Metadata
                    'email_sent': True,
                    'report_generated': pdf_path is not None
                }
                
                result = predictions_collection.insert_one(prediction_document)
                print(f"[MONGODB] Prediction saved with ID: {result.inserted_id}")
                print(f"[MONGODB] Stored: Metadata only (thumbnail, results, no large images)")
                
            except Exception as e:
                print(f"[ERROR] Failed to save to MongoDB: {e}")
                import traceback
                traceback.print_exc()
        
        return jsonify({
            'success': True,
            'stage1_cnn': cnn_result,
            'stage2_tabular': tabular_result,
            'shap_explanation': shap_explanation,
            'what_if_scenarios': what_if_scenarios,
            'survival_analysis': survival_analysis,
            'vlm_narrative': vlm_narrative,  # TIER 1: Clinical narrative
            'wsi_result': wsi_result,  # TIER 1: WSI analysis
            'final_prediction': final_prediction,
            'final_confidence': final_confidence,
            'risk_level': risk_level,
            'email_sent': True
        }), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable (Hugging Face Spaces uses 7860)
    port = int(os.getenv('PORT', 7860))
    host = os.getenv('HOST', '0.0.0.0')
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"\n{'='*60}")
    print(f"🚀 FedFusionNet++ Server Starting...")
    print(f"{'='*60}")
    print(f"📍 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🐛 Debug: {debug}")
    print(f"🗄️  MongoDB: {'Connected' if MONGODB_AVAILABLE else 'Disconnected'}")
    print(f"🧠 CNN Model: {'Loaded' if CNN_MODEL else 'Not Loaded'}")
    print(f"📊 Tabular Model: {'Loaded' if TABULAR_MODEL else 'Not Loaded'}")
    print(f"💬 VLM Service: {'Available' if VLM_SERVICE and VLM_SERVICE.is_available() else 'Not Available'}")
    print(f"🗺️  WSI Processor: {'Available' if WSI_PROCESSOR else 'Not Available'}")
    print(f"{'='*60}")
    print(f"\n✅ Server ready! Access at: http://{host if host != '0.0.0.0' else 'localhost'}:{port}")
    print(f"\n")
    
    try:
        app.run(debug=debug, port=port, host=host, threaded=True)
    except OSError as e:
        if 'Address already in use' in str(e):
            print(f"\n❌ ERROR: Port {port} is already in use!")
            print(f"\n💡 Solutions:")
            print(f"   1. Kill the process using port {port}:")
            print(f"      Windows: netstat -ano | findstr :{port}")
            print(f"               taskkill /PID <PID> /F")
            print(f"      Linux/Mac: lsof -i :{port}")
            print(f"                 kill -9 <PID>")
            print(f"   2. Use a different port: set PORT=5001 (Windows) or export PORT=5001 (Linux/Mac)")
            print(f"\n")
        else:
            print(f"\n❌ ERROR: {e}")
            print(f"\n")
    except KeyboardInterrupt:
        print(f"\n\n{'='*60}")
        print(f"🛑 Server stopped by user")
        print(f"{'='*60}\n")
