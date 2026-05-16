"""
SHAP Explainability Module for NeuroPlex
Provides risk factor contribution analysis
"""

import numpy as np
import shap
from pathlib import Path
import pickle

class RiskFactorExplainer:
    """
    Explains how each patient risk factor contributes to the final prediction
    """
    
    def __init__(self, tabular_model_path=None):
        """
        Initialize SHAP explainer for tabular model
        
        Args:
            tabular_model_path: Path to stage2_tabular_model.pkl
        """
        self.explainer = None
        self.feature_names = [
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
            'Country',
            'cnn_diagnosis',
            'cnn_probability',
            'cnn_confidence',
            'cnn_uncertainty'
        ]
        
        if tabular_model_path:
            self.load_explainer(tabular_model_path)
    
    def load_explainer(self, model_path):
        """
        Load the tabular model and create SHAP explainer
        
        Args:
            model_path: Path to the trained model
        """
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Create SHAP explainer for tree-based model
            self.explainer = shap.TreeExplainer(model)
            print("[SHAP] Explainer loaded successfully")
            
        except Exception as e:
            print(f"[SHAP] Warning: Could not load explainer: {e}")
            self.explainer = None
    
    def explain_prediction(self, patient_data, cnn_outputs):
        """
        Generate SHAP explanation for a single prediction
        
        Args:
            patient_data: Dictionary of patient features
            cnn_outputs: Dictionary of CNN model outputs
            
        Returns:
            Dictionary containing SHAP values and contribution analysis
        """
        if self.explainer is None:
            return self._fallback_explanation(patient_data, cnn_outputs)
        
        try:
            # Prepare feature vector
            features = self._prepare_features(patient_data, cnn_outputs)
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(features)
            
            # Get base value (expected value)
            base_value = self.explainer.expected_value
            
            # If multi-class, take OSCC class (index 1)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # OSCC class
                if isinstance(base_value, (list, np.ndarray)):
                    base_value = base_value[1]
            
            # Create contribution analysis
            contributions = self._analyze_contributions(
                features[0], 
                shap_values[0] if len(shap_values.shape) > 1 else shap_values,
                base_value
            )
            
            return contributions
            
        except Exception as e:
            print(f"[SHAP] Error in explanation: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_explanation(patient_data, cnn_outputs)
    
    def _prepare_features(self, patient_data, cnn_outputs):
        """
        Convert patient data and CNN outputs to feature vector
        
        Args:
            patient_data: Dictionary of patient features
            cnn_outputs: Dictionary of CNN outputs
            
        Returns:
            numpy array of shape (1, 20)
        """
        # Encode categorical features
        feature_vector = []
        
        # Age (numeric)
        feature_vector.append(float(patient_data.get('Age', 45)))
        
        # Gender (binary)
        feature_vector.append(1.0 if patient_data.get('Gender', 'Male') == 'Male' else 0.0)
        
        # Binary yes/no features
        binary_features = [
            'Tobacco Use',
            'Alcohol Consumption',
            'HPV Infection',
            'Betel Quid Use',
            'Chronic Sun Exposure',
            'Poor Oral Hygiene',
            'Family History of Cancer',
            'Compromised Immune System',
            'Oral Lesions',
            'Unexplained Bleeding',
            'Difficulty Swallowing',
            'White or Red Patches in Mouth'
        ]
        
        for feature in binary_features:
            value = patient_data.get(feature, 'No')
            feature_vector.append(1.0 if value == 'Yes' else 0.0)
        
        # Diet (ordinal: Low=0, Medium=1, High=2)
        diet = patient_data.get('Diet (Fruits & Vegetables Intake)', 'Low')
        diet_map = {'Low': 0.0, 'Medium': 1.0, 'High': 2.0}
        feature_vector.append(diet_map.get(diet, 0.0))
        
        # Country (encoded as numeric - simplified)
        country = patient_data.get('Country', 'India')
        country_map = {'India': 0.0, 'USA': 1.0, 'UK': 2.0, 'Other': 3.0}
        feature_vector.append(country_map.get(country, 0.0))
        
        # CNN outputs
        feature_vector.append(float(cnn_outputs.get('cnn_diagnosis', 0)))
        feature_vector.append(float(cnn_outputs.get('cnn_probability', 0)))
        feature_vector.append(float(cnn_outputs.get('cnn_confidence', 0)))
        feature_vector.append(float(cnn_outputs.get('cnn_uncertainty', 0)))
        
        return np.array(feature_vector).reshape(1, -1)
    
    def _analyze_contributions(self, features, shap_values, base_value):
        """
        Analyze SHAP values to create human-readable contributions
        
        Args:
            features: Feature vector
            shap_values: SHAP values for each feature
            base_value: Base prediction value
            
        Returns:
            Dictionary with contribution analysis
        """
        # Create contribution list
        contributions = []
        
        for i, (feature_name, feature_value, shap_value) in enumerate(
            zip(self.feature_names, features, shap_values)
        ):
            # Skip if contribution is negligible
            if abs(shap_value) < 0.001:
                continue
            
            contribution = {
                'feature': feature_name,
                'value': self._format_feature_value(feature_name, feature_value),
                'contribution': float(shap_value),
                'contribution_percent': float(shap_value / (abs(shap_values).sum() + 1e-10) * 100),
                'direction': 'increases' if shap_value > 0 else 'decreases'
            }
            contributions.append(contribution)
        
        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        # Calculate cumulative risk
        cumulative_risk = base_value
        waterfall_data = [{'feature': 'Base Risk', 'value': base_value, 'cumulative': base_value}]
        
        for contrib in contributions[:10]:  # Top 10 contributors
            cumulative_risk += contrib['contribution']
            waterfall_data.append({
                'feature': contrib['feature'],
                'value': contrib['contribution'],
                'cumulative': cumulative_risk
            })
        
        return {
            'base_risk': float(base_value),
            'final_risk': float(cumulative_risk),
            'contributions': contributions[:15],  # Top 15 factors
            'waterfall_data': waterfall_data,
            'top_risk_factors': [c for c in contributions if c['contribution'] > 0][:5],
            'top_protective_factors': [c for c in contributions if c['contribution'] < 0][:5]
        }
    
    def _format_feature_value(self, feature_name, value):
        """
        Format feature value for display
        
        Args:
            feature_name: Name of the feature
            value: Numeric value
            
        Returns:
            Formatted string
        """
        if feature_name == 'Age':
            return f"{int(value)} years"
        elif feature_name == 'Gender':
            return 'Male' if value > 0.5 else 'Female'
        elif feature_name == 'Diet (Fruits & Vegetables Intake)':
            diet_map = {0.0: 'Low', 1.0: 'Medium', 2.0: 'High'}
            return diet_map.get(value, 'Unknown')
        elif feature_name in ['cnn_diagnosis', 'cnn_probability', 'cnn_confidence', 'cnn_uncertainty']:
            return f"{value:.3f}"
        else:
            return 'Yes' if value > 0.5 else 'No'
    
    def _fallback_explanation(self, patient_data, cnn_outputs):
        """
        Provide rule-based explanation when SHAP is unavailable
        
        Args:
            patient_data: Dictionary of patient features
            cnn_outputs: Dictionary of CNN outputs
            
        Returns:
            Dictionary with basic contribution analysis
        """
        contributions = []
        
        # Rule-based risk scoring
        risk_weights = {
            'Tobacco Use': 0.25,
            'Betel Quid Use': 0.20,
            'Alcohol Consumption': 0.15,
            'HPV Infection': 0.10,
            'Poor Oral Hygiene': 0.08,
            'Family History of Cancer': 0.07,
            'Oral Lesions': 0.05,
            'White or Red Patches in Mouth': 0.05,
            'Unexplained Bleeding': 0.03,
            'Difficulty Swallowing': 0.02
        }
        
        for feature, weight in risk_weights.items():
            if patient_data.get(feature) == 'Yes':
                contributions.append({
                    'feature': feature,
                    'value': 'Yes',
                    'contribution': weight,
                    'contribution_percent': weight * 100,
                    'direction': 'increases'
                })
        
        # Add CNN contribution
        cnn_contrib = cnn_outputs.get('cnn_probability', 0) * 0.5
        contributions.append({
            'feature': 'CNN Image Analysis',
            'value': f"{cnn_outputs.get('cnn_probability', 0):.2f}",
            'contribution': cnn_contrib,
            'contribution_percent': cnn_contrib * 100,
            'direction': 'increases' if cnn_contrib > 0 else 'decreases'
        })
        
        contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        return {
            'base_risk': 0.1,
            'final_risk': sum(c['contribution'] for c in contributions) + 0.1,
            'contributions': contributions,
            'waterfall_data': [],
            'top_risk_factors': contributions[:5],
            'top_protective_factors': [],
            'note': 'Using rule-based explanation (SHAP unavailable)'
        }
    
    def generate_what_if_scenarios(self, patient_data, cnn_outputs):
        """
        Generate what-if scenarios showing impact of lifestyle changes
        
        Args:
            patient_data: Dictionary of patient features
            cnn_outputs: Dictionary of CNN outputs
            
        Returns:
            List of scenario dictionaries
        """
        scenarios = []
        
        # Scenario 1: Stop tobacco
        if patient_data.get('Tobacco Use') == 'Yes':
            modified_data = patient_data.copy()
            modified_data['Tobacco Use'] = 'No'
            scenarios.append({
                'name': 'Stop Tobacco Use',
                'description': 'If patient stops using tobacco',
                'modifiable': True,
                'impact': 'High',
                'risk_reduction': '25-35%'
            })
        
        # Scenario 2: Stop betel quid
        if patient_data.get('Betel Quid Use') == 'Yes':
            scenarios.append({
                'name': 'Stop Betel Quid',
                'description': 'If patient stops chewing betel quid',
                'modifiable': True,
                'impact': 'High',
                'risk_reduction': '20-30%'
            })
        
        # Scenario 3: Stop alcohol
        if patient_data.get('Alcohol Consumption') == 'Yes':
            scenarios.append({
                'name': 'Stop Alcohol',
                'description': 'If patient stops alcohol consumption',
                'modifiable': True,
                'impact': 'Medium',
                'risk_reduction': '15-20%'
            })
        
        # Scenario 4: Improve oral hygiene
        if patient_data.get('Poor Oral Hygiene') == 'Yes':
            scenarios.append({
                'name': 'Improve Oral Hygiene',
                'description': 'Regular brushing, flossing, dental checkups',
                'modifiable': True,
                'impact': 'Medium',
                'risk_reduction': '8-12%'
            })
        
        # Scenario 5: Improve diet
        if patient_data.get('Diet (Fruits & Vegetables Intake)') == 'Low':
            scenarios.append({
                'name': 'Improve Diet',
                'description': 'Increase fruits and vegetables intake',
                'modifiable': True,
                'impact': 'Low-Medium',
                'risk_reduction': '5-10%'
            })
        
        return scenarios


def create_shap_explainer(model_path=None):
    """
    Factory function to create SHAP explainer
    
    Args:
        model_path: Path to tabular model
        
    Returns:
        RiskFactorExplainer instance
    """
    if model_path is None:
        model_path = Path(__file__).parent.parent / 'models' / 'stage2_tabular_model.pkl'
    
    return RiskFactorExplainer(str(model_path))
