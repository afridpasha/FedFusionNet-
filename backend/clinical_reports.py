"""
FedFusionNet++ - Clinical Report Generator
Generates PDF reports and HL7 FHIR R4 compliant JSON
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from datetime import datetime
import json
import io
from PIL import Image
import numpy as np

# ============================================================================
# PDF REPORT GENERATOR
# ============================================================================

class PDFReportGenerator:
    """
    Generate professional PDF diagnostic reports
    """
    
    def __init__(self):
        self.page_width, self.page_height = letter
    
    def generate_report(self, output_path, patient_info, prediction_results, xai_results, heatmap_images=None):
        """
        Generate comprehensive PDF report
        
        Args:
            output_path: Path to save PDF
            patient_info: Dict with patient details
            prediction_results: Dict with model predictions
            xai_results: Dict with XAI explanations
            heatmap_images: Dict with base64 encoded heatmap images (gradcam, scorecam)
        
        Returns:
            output_path: Path to generated PDF
        """
        c = canvas.Canvas(output_path, pagesize=letter)
        
        # Header
        self._draw_header(c)
        
        # Patient Information
        y_pos = self._draw_patient_info(c, patient_info, start_y=self.page_height - 120)
        
        # Prediction Results
        y_pos = self._draw_prediction_results(c, prediction_results, start_y=y_pos - 40)
        
        # Risk Assessment
        y_pos = self._draw_risk_assessment(c, xai_results, start_y=y_pos - 40)
        
        # XAI Heatmaps (if available)
        if heatmap_images and y_pos > 300:
            y_pos = self._draw_heatmaps(c, heatmap_images, start_y=y_pos - 40)
        elif heatmap_images:
            # Start new page for heatmaps if not enough space
            c.showPage()
            y_pos = self._draw_heatmaps(c, heatmap_images, start_y=self.page_height - 80)
        
        # Model Information
        self._draw_model_info(c, xai_results)
        
        # Footer
        self._draw_footer(c)
        
        c.save()
        return output_path
    
    def _draw_header(self, c):
        """Draw report header"""
        c.setFont("Helvetica-Bold", 20)
        c.drawString(50, self.page_height - 50, "FedFusionNet++ Diagnostic Report")
        
        c.setFont("Helvetica", 10)
        c.drawString(50, self.page_height - 70, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Draw line
        c.line(50, self.page_height - 80, self.page_width - 50, self.page_height - 80)
    
    def _draw_patient_info(self, c, patient_info, start_y):
        """Draw patient information section"""
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, start_y, "Patient Information")
        
        c.setFont("Helvetica", 11)
        y = start_y - 25
        
        info_items = [
            ("Patient ID:", patient_info.get('patient_id', 'N/A')),
            ("Age:", str(patient_info.get('age', 'N/A'))),
            ("Gender:", patient_info.get('gender', 'N/A')),
            ("Hospital ID:", patient_info.get('hospital_id', 'N/A'))
        ]
        
        for label, value in info_items:
            c.drawString(50, y, f"{label}")
            c.drawString(200, y, str(value))
            y -= 20
        
        return y
    
    def _draw_prediction_results(self, c, prediction_results, start_y):
        """Draw prediction results section"""
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, start_y, "Classification Results")
        
        y = start_y - 25
        
        # Main prediction
        pred_class = prediction_results.get('class', 0)
        label = 'OSCC (Oral Squamous Cell Carcinoma)' if pred_class == 1 else 'Normal Tissue'
        confidence = prediction_results.get('confidence', 0.0)
        
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Diagnosis:")
        c.setFont("Helvetica", 12)
        
        # Color code based on prediction
        if pred_class == 1:
            c.setFillColorRGB(0.8, 0.2, 0.2)  # Red for OSCC
        else:
            c.setFillColorRGB(0.2, 0.6, 0.2)  # Green for Normal
        
        c.drawString(200, y, label)
        c.setFillColorRGB(0, 0, 0)  # Reset to black
        
        y -= 25
        
        # Confidence and uncertainty
        c.setFont("Helvetica", 11)
        c.drawString(50, y, f"Confidence: {confidence:.2%}")
        y -= 20
        
        uncertainty = prediction_results.get('uncertainty', 0.0)
        c.drawString(50, y, f"Uncertainty: {uncertainty:.4f}")
        y -= 20
        
        # Stage information (if available)
        if 'stage' in prediction_results:
            stage = prediction_results['stage']
            c.drawString(50, y, f"Cancer Stage: {stage}")
            y -= 20
        
        return y
    
    def _draw_risk_assessment(self, c, xai_results, start_y):
        """Draw risk assessment section"""
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, start_y, "Risk Stratification")
        
        y = start_y - 25
        
        risk_info = xai_results.get('risk', {})
        tier = risk_info.get('tier', 'UNKNOWN')
        action = risk_info.get('action', 'N/A')
        color_hex = risk_info.get('color', '#000000')
        
        # Convert hex to RGB
        color_rgb = tuple(int(color_hex[i:i+2], 16)/255 for i in (1, 3, 5))
        
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Risk Tier:")
        c.setFillColorRGB(*color_rgb)
        c.drawString(200, y, tier)
        c.setFillColorRGB(0, 0, 0)
        
        y -= 25
        
        c.setFont("Helvetica", 11)
        c.drawString(50, y, "Recommended Action:")
        y -= 20
        
        # Wrap action text if too long
        max_width = 500
        words = action.split()
        line = ""
        for word in words:
            test_line = line + word + " "
            if c.stringWidth(test_line, "Helvetica", 11) < max_width:
                line = test_line
            else:
                c.drawString(70, y, line)
                y -= 15
                line = word + " "
        if line:
            c.drawString(70, y, line)
            y -= 20
        
        return y
    
    def _draw_heatmaps(self, c, heatmap_images, start_y):
        """Draw XAI heatmap images"""
        import base64
        from io import BytesIO
        from PIL import Image as PILImage
        
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, start_y, "Explainable AI (XAI) - Visual Explanations")
        
        y = start_y - 20
        c.setFont("Helvetica", 10)
        c.drawString(50, y, "Heatmaps showing which regions influenced the AI's decision")
        
        y -= 30
        
        # Image dimensions
        img_width = 240
        img_height = 240
        
        # Draw Grad-CAM++ heatmap
        if 'gradcam' in heatmap_images and heatmap_images['gradcam']:
            try:
                # Decode base64 image
                gradcam_data = heatmap_images['gradcam']
                if gradcam_data.startswith('data:image'):
                    gradcam_data = gradcam_data.split(',')[1]
                
                img_data = base64.b64decode(gradcam_data)
                img = PILImage.open(BytesIO(img_data))
                
                # Draw image
                c.drawImage(ImageReader(img), 50, y - img_height, width=img_width, height=img_height)
                
                # Label
                c.setFont("Helvetica-Bold", 11)
                c.drawString(50, y - img_height - 15, "Grad-CAM++ Heatmap")
                c.setFont("Helvetica", 9)
                c.drawString(50, y - img_height - 28, "Red/Yellow: High diagnostic importance")
                
            except Exception as e:
                print(f"[PDF] Error drawing Grad-CAM++ heatmap: {e}")
                c.setFont("Helvetica", 10)
                c.drawString(50, y, "Grad-CAM++ heatmap unavailable")
        
        # Draw Layer-CAM heatmap (next to Grad-CAM++)
        if 'scorecam' in heatmap_images and heatmap_images['scorecam']:
            try:
                # Decode base64 image
                scorecam_data = heatmap_images['scorecam']
                if scorecam_data.startswith('data:image'):
                    scorecam_data = scorecam_data.split(',')[1]
                
                img_data = base64.b64decode(scorecam_data)
                img = PILImage.open(BytesIO(img_data))
                
                # Draw image (next to first image)
                x_offset = 320
                c.drawImage(ImageReader(img), x_offset, y - img_height, width=img_width, height=img_height)
                
                # Label
                c.setFont("Helvetica-Bold", 11)
                c.drawString(x_offset, y - img_height - 15, "Layer-CAM Heatmap")
                c.setFont("Helvetica", 9)
                c.drawString(x_offset, y - img_height - 28, "Alternative visualization for validation")
                
            except Exception as e:
                print(f"[PDF] Error drawing Layer-CAM heatmap: {e}")
        
        # Return new y position
        return y - img_height - 40
    
    def _draw_model_info(self, c, xai_results):
        """Draw model information at bottom"""
        y = 100
        
        c.setFont("Helvetica-Bold", 10)
        c.drawString(50, y, "Model Information")
        
        c.setFont("Helvetica", 9)
        y -= 15
        
        metadata = xai_results.get('metadata', {})
        model_name = metadata.get('model', 'FedFusionNet++ v2.0')
        mc_passes = metadata.get('mc_dropout_passes', 50)
        
        c.drawString(50, y, f"Model: {model_name}")
        y -= 12
        c.drawString(50, y, f"Architecture: Swin-ViT + CrossViT with SE Fusion")
        y -= 12
        c.drawString(50, y, f"Uncertainty Estimation: MC-Dropout ({mc_passes} passes)")
        y -= 12
        c.drawString(50, y, f"Privacy: Differential Privacy (ε=3.0, δ=1e-5)")
    
    def _draw_footer(self, c):
        """Draw report footer"""
        c.setFont("Helvetica", 8)
        c.drawString(50, 30, "This report is generated by AI and should be reviewed by a qualified pathologist.")
        c.drawString(50, 20, "© 2024 FedFusionNet++ - AI-Powered Oral Cancer Detection System")


# ============================================================================
# HL7 FHIR R4 JSON GENERATOR
# ============================================================================

class FHIRReportGenerator:
    """
    Generate HL7 FHIR R4 compliant DiagnosticReport JSON
    """
    
    def __init__(self):
        pass
    
    def generate_report(self, patient_info, prediction_results, xai_results):
        """
        Generate FHIR R4 DiagnosticReport
        
        Args:
            patient_info: Dict with patient details
            prediction_results: Dict with model predictions
            xai_results: Dict with XAI explanations
        
        Returns:
            dict: FHIR R4 compliant JSON
        """
        patient_id = patient_info.get('patient_id', 'unknown')
        pred_class = prediction_results.get('class', 0)
        label = 'OSCC' if pred_class == 1 else 'Normal'
        confidence = prediction_results.get('confidence', 0.0)
        uncertainty = prediction_results.get('uncertainty', 0.0)
        
        risk_info = xai_results.get('risk', {})
        risk_tier = risk_info.get('tier', 'UNKNOWN')
        
        # Create FHIR DiagnosticReport
        fhir_report = {
            "resourceType": "DiagnosticReport",
            "id": f"fedfusion-{patient_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                    "code": "PAT",
                    "display": "Pathology"
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "11529-5",
                    "display": "Surgical pathology study"
                }],
                "text": "Oral Cancer AI Screening"
            },
            "subject": {
                "reference": f"Patient/{patient_id}",
                "display": f"Patient {patient_id}"
            },
            "effectiveDateTime": datetime.now().isoformat(),
            "issued": datetime.now().isoformat(),
            "performer": [{
                "reference": "Organization/fedfusionnet",
                "display": "FedFusionNet++ AI System"
            }],
            "conclusion": f"{label} - Risk Tier: {risk_tier} (Confidence: {confidence:.2%}, Uncertainty: {uncertainty:.4f})",
            "conclusionCode": [{
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": "363346000" if pred_class == 1 else "260385009",
                    "display": "Malignant neoplasm" if pred_class == 1 else "Negative finding"
                }]
            }],
            "result": [
                self._create_observation(
                    "confidence",
                    "AI Model Confidence",
                    confidence,
                    "%"
                ),
                self._create_observation(
                    "uncertainty",
                    "Prediction Uncertainty",
                    uncertainty,
                    "score"
                ),
                self._create_observation(
                    "risk-tier",
                    "Clinical Risk Tier",
                    risk_tier,
                    None
                )
            ]
        }
        
        # Add stage if available
        if 'stage' in prediction_results:
            stage = prediction_results['stage']
            fhir_report['result'].append(
                self._create_observation(
                    "cancer-stage",
                    "Cancer Stage",
                    f"Stage {stage}",
                    None
                )
            )
        
        return fhir_report
    
    def _create_observation(self, obs_id, display, value, unit):
        """Create FHIR Observation resource"""
        observation = {
            "reference": f"Observation/{obs_id}",
            "display": display,
            "resource": {
                "resourceType": "Observation",
                "id": obs_id,
                "status": "final",
                "code": {
                    "text": display
                },
                "effectiveDateTime": datetime.now().isoformat()
            }
        }
        
        if isinstance(value, (int, float)):
            observation['resource']['valueQuantity'] = {
                "value": value,
                "unit": unit
            }
        else:
            observation['resource']['valueString'] = str(value)
        
        return observation
    
    def save_to_file(self, fhir_report, output_path):
        """Save FHIR report to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(fhir_report, f, indent=2)
        return output_path


# ============================================================================
# COMPLETE CLINICAL REPORT GENERATOR
# ============================================================================

class ClinicalReportGenerator:
    """
    Complete clinical report generation system
    Generates both PDF and FHIR JSON reports
    """
    
    def __init__(self):
        self.pdf_generator = PDFReportGenerator()
        self.fhir_generator = FHIRReportGenerator()
    
    def generate_pdf_report(self, complete_result, output_dir='reports'):
        """
        Generate PDF report from complete result data
        
        Args:
            complete_result: Dict with all prediction data
            output_dir: Directory to save report
        
        Returns:
            str: Path to generated PDF
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        patient_id = complete_result.get('patient_id', 'unknown')
        
        # Prepare patient info
        patient_info = {
            'patient_id': patient_id,
            'age': complete_result.get('patient_data', {}).get('Age', 'N/A'),
            'gender': complete_result.get('patient_data', {}).get('Gender', 'N/A'),
            'hospital_id': complete_result.get('hospital_id', 'N/A')
        }
        
        # Prepare prediction results
        stage1 = complete_result.get('stage1_cnn', {})
        stage2 = complete_result.get('stage2_tabular', {})
        
        prediction_results = {
            'class': 1 if complete_result.get('final_prediction') == 'OSCC' else 0,
            'confidence': complete_result.get('final_confidence', 0) / 100,
            'uncertainty': stage1.get('uncertainty', 0),
            'stage': stage2.get('cancer_stage', 'N/A') if stage2 else 'N/A'
        }
        
        # Prepare XAI results
        xai_data = stage1.get('xai', {})
        xai_results = {
            'risk': {
                'tier': xai_data.get('risk_tier', complete_result.get('risk_level', 'UNKNOWN')),
                'action': xai_data.get('risk_action', 'Consult with specialist'),
                'color': xai_data.get('risk_color', '#667eea')
            },
            'metadata': {
                'model': 'FedFusionNet++ v2.0',
                'mc_dropout_passes': 50
            }
        }
        
        # Extract heatmap images
        heatmap_images = None
        if xai_data:
            heatmap_images = {
                'gradcam': xai_data.get('gradcam'),
                'scorecam': xai_data.get('scorecam')
            }
        
        # Generate PDF
        pdf_path = os.path.join(output_dir, f'report_{patient_id}_{timestamp}.pdf')
        self.pdf_generator.generate_report(pdf_path, patient_info, prediction_results, xai_results, heatmap_images)
        
        return pdf_path
    
    def generate_json_report(self, complete_result, output_dir='reports'):
        """
        Generate JSON report from complete result data
        
        Args:
            complete_result: Dict with all prediction data
            output_dir: Directory to save report
        
        Returns:
            str: Path to generated JSON
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        patient_id = complete_result.get('patient_id', 'unknown')
        
        # Prepare patient info
        patient_info = {
            'patient_id': patient_id,
            'age': complete_result.get('patient_data', {}).get('Age', 'N/A'),
            'gender': complete_result.get('patient_data', {}).get('Gender', 'N/A'),
            'hospital_id': complete_result.get('hospital_id', 'N/A')
        }
        
        # Prepare prediction results
        stage1 = complete_result.get('stage1_cnn', {})
        stage2 = complete_result.get('stage2_tabular', {})
        
        prediction_results = {
            'class': 1 if complete_result.get('final_prediction') == 'OSCC' else 0,
            'confidence': complete_result.get('final_confidence', 0) / 100,
            'uncertainty': stage1.get('uncertainty', 0),
            'stage': stage2.get('cancer_stage', 'N/A') if stage2 else 'N/A'
        }
        
        # Prepare XAI results
        xai_data = stage1.get('xai', {})
        xai_results = {
            'risk': {
                'tier': xai_data.get('risk_tier', complete_result.get('risk_level', 'UNKNOWN')),
                'action': xai_data.get('risk_action', 'Consult with specialist'),
                'color': xai_data.get('risk_color', '#667eea')
            },
            'metadata': {
                'model': 'FedFusionNet++ v2.0',
                'mc_dropout_passes': 50
            }
        }
        
        # Generate FHIR JSON
        fhir_report = self.fhir_generator.generate_report(patient_info, prediction_results, xai_results)
        json_path = os.path.join(output_dir, f'report_{patient_id}_{timestamp}.json')
        self.fhir_generator.save_to_file(fhir_report, json_path)
        
        return json_path
    
    def generate_complete_report(self, patient_info, prediction_results, xai_results, 
                                 output_dir='reports'):
        """
        Generate both PDF and FHIR reports
        
        Args:
            patient_info: Dict with patient details
            prediction_results: Dict with model predictions
            xai_results: Dict with XAI explanations
            output_dir: Directory to save reports
        
        Returns:
            dict with paths to generated reports
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        patient_id = patient_info.get('patient_id', 'unknown')
        
        # Generate PDF
        pdf_path = os.path.join(output_dir, f'report_{patient_id}_{timestamp}.pdf')
        self.pdf_generator.generate_report(pdf_path, patient_info, prediction_results, xai_results)
        
        # Generate FHIR JSON
        fhir_report = self.fhir_generator.generate_report(patient_info, prediction_results, xai_results)
        fhir_path = os.path.join(output_dir, f'report_{patient_id}_{timestamp}.json')
        self.fhir_generator.save_to_file(fhir_report, fhir_path)
        
        return {
            'pdf': pdf_path,
            'fhir_json': fhir_path,
            'timestamp': timestamp
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    print("FedFusionNet++ Clinical Report Generator")
    print("="*80)
    print("\nFeatures:")
    print("  1. PDF Report Generation")
    print("     - Professional medical report format")
    print("     - Patient information")
    print("     - Prediction results with confidence")
    print("     - Risk stratification")
    print("     - Model metadata")
    print("\n  2. HL7 FHIR R4 JSON Generation")
    print("     - DiagnosticReport resource")
    print("     - Observation resources")
    print("     - SNOMED CT coding")
    print("     - LOINC coding")
    print("\nUsage:")
    print("  from backend.clinical_reports import ClinicalReportGenerator")
    print("  generator = ClinicalReportGenerator()")
    print("  reports = generator.generate_complete_report(patient_info, prediction, xai)")
