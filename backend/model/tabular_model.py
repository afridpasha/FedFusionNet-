"""
Stage-2 Tabular Model for Clinical Data
Uses 16 patient features + 4 CNN outputs (from Stage-1) to predict:
- Cancer Stage (0-4)
- Survival Rate (5-year %)
- Treatment Type
- Cost of Treatment
- Economic Burden
"""

import numpy as np
import joblib
from pathlib import Path

class OralCancerTabularModel:
    def __init__(self, model_path=None):
        """
        Initialize Stage-2 Tabular Model
        
        Args:
            model_path: Path to pretrained model (stage2_tabular_model.pkl)
        """
        self.model = None
        
        # 16 patient features (user input)
        self.patient_features = [
            'Age',
            'Gender',
            'Tobacco Use',
            'Alcohol Consumption',
            'HPV Infection',
            'Betel Quid Use',
            'Chronic Sun Exposure',
            'Poor Oral Hygiene',
            'Diet (Fruits & Vegetables Intake)',
            'Family History of Cancer',
            'Compromised Immune System',
            'Oral Lesions',
            'Unexplained Bleeding',
            'Difficulty Swallowing',
            'White or Red Patches in Mouth',
            'Country'
        ]
        
        # 4 CNN output features (from Stage-1)
        self.cnn_features = [
            'cnn_diagnosis',      # 0 or 1
            'cnn_probability',    # 0.01-0.99
            'cnn_confidence',     # 0/1/2 (Low/Moderate/High)
            'cnn_uncertainty'     # 0.01-0.35
        ]
        
        # Load pretrained model
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Load pretrained Stage-2 model
        Uses custom wrapper for compatibility
        
        Args:
            model_path: Path to stage2_tabular_model.pkl
        """
        try:
            model_path = Path(model_path)
            if model_path.exists():
                # Try to load with joblib first
                try:
                    self.model = joblib.load(str(model_path))
                    print(f"[OK] Stage-2 model loaded: {model_path.name}")
                except (ModuleNotFoundError, AttributeError) as e:
                    print(f"[INFO] Using custom wrapper for Stage-2 model")
                    # Use custom wrapper for old TabPFN models
                    from backend.model.custom_tabpfn_wrapper import CustomTabPFNWrapper
                    self.model = CustomTabPFNWrapper(str(model_path))
                    print(f"[OK] Custom wrapper loaded successfully")
            else:
                print(f"[ERROR] Model not found: {model_path}")
        except Exception as e:
            print(f"[ERROR] Error loading model: {e}")
    
    def prepare_features(self, patient_data, cnn_outputs):
        """
        Prepare feature vector for prediction
        
        Args:
            patient_data: dict with 16 patient features
            cnn_outputs: dict with 4 CNN outputs from Stage-1
        
        Returns:
            features: numpy array of 20 features (all numerical)
        """
        features = []
        
        # Add 16 patient features (convert to numerical)
        for feature_name in self.patient_features:
            value = patient_data.get(feature_name, 'No' if feature_name != 'Age' else 45)
            
            # Convert to numerical
            if feature_name == 'Age':
                features.append(float(value))
            elif feature_name == 'Gender':
                features.append(1.0 if value == 'Male' else 0.0)
            elif feature_name == 'Diet (Fruits & Vegetables Intake)':
                diet_map = {'Low': 0.0, 'Medium': 0.5, 'High': 1.0}
                features.append(diet_map.get(value, 0.0))
            elif feature_name == 'Country':
                # Simple encoding: India=1, others=0
                features.append(1.0 if value == 'India' else 0.0)
            else:
                # Binary features (Yes/No)
                features.append(1.0 if value == 'Yes' else 0.0)
        
        # Add 4 CNN features from Stage-1
        features.append(float(cnn_outputs.get('cnn_diagnosis', 0)))
        features.append(float(cnn_outputs.get('cnn_probability', 0.5)))
        features.append(float(cnn_outputs.get('cnn_confidence', 0)))
        features.append(float(cnn_outputs.get('cnn_uncertainty', 0.1)))
        
        return np.array(features, dtype=np.float64).reshape(1, -1)
    
    def predict(self, patient_data, cnn_outputs):
        """
        Predict cancer stage and related outputs
        
        Args:
            patient_data: dict with 16 patient features
            cnn_outputs: dict with 4 CNN outputs from Stage-1
                - cnn_diagnosis: 0 or 1
                - cnn_probability: 0.01-0.99
                - cnn_confidence: 0/1/2
                - cnn_uncertainty: 0.01-0.35
        
        Returns:
            dict with:
                - cancer_stage: int (0-4)
                - stage_confidence: float (0-1)
                - survival_rate_5yr: float (percentage)
                - treatment_type: str
                - cost_usd: float
                - economic_burden_days: int
        """
        if self.model is None:
            # Fallback if model not loaded
            return {
                'cancer_stage': 0,
                'stage_confidence': 0.5,
                'survival_rate_5yr': 100.0,
                'treatment_type': 'No Treatment',
                'cost_usd': 0.0,
                'economic_burden_days': 0
            }
        
        try:
            # Prepare features (16 patient + 4 CNN = 20 total)
            features = self.prepare_features(patient_data, cnn_outputs)
            
            # Predict using pretrained model
            prediction = self.model.predict(features)[0]
            
            # Get prediction probabilities if available
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features)[0]
                confidence = float(proba.max())
            else:
                confidence = 0.85  # Default confidence
            
            # Map stage to outputs (based on your CSV dataset)
            stage_mapping = {
                0: {'survival': 100.0, 'treatment': 'No Treatment', 'cost': 0.0, 'burden': 0},
                1: {'survival': 85.0, 'treatment': 'Surgery', 'cost': 75000.0, 'burden': 90},
                2: {'survival': 68.0, 'treatment': 'Surgery + Radiation', 'cost': 95000.0, 'burden': 130},
                3: {'survival': 45.0, 'treatment': 'Chemotherapy + Radiation', 'cost': 120000.0, 'burden': 180},
                4: {'survival': 20.0, 'treatment': 'Palliative Care', 'cost': 150000.0, 'burden': 200}
            }
            
            stage = int(prediction)
            stage_info = stage_mapping.get(stage, stage_mapping[0])
            
            return {
                'cancer_stage': stage,
                'stage_confidence': confidence,
                'survival_rate_5yr': stage_info['survival'],
                'treatment_type': stage_info['treatment'],
                'cost_usd': stage_info['cost'],
                'economic_burden_days': stage_info['burden']
            }
            
        except Exception as e:
            print(f"[ERROR] Prediction error: {e}")
            return {
                'cancer_stage': 0,
                'stage_confidence': 0.5,
                'survival_rate_5yr': 100.0,
                'treatment_type': 'No Treatment',
                'cost_usd': 0.0,
                'economic_burden_days': 0
            }
