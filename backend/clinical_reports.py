"""
NeuroPlex - Clinical Report Generator
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
    
    def generate_report(self, output_path, patient_info, prediction_results, xai_results, heatmap_images=None, wsi_data=None, survival_data=None):
        """
        Generate comprehensive PDF report with ALL sections (IN-MEMORY, NO TEMP FILES)
        
        Args:
            output_path: Path to save PDF
            patient_info: Dict with patient details
            prediction_results: Dict with model predictions
            xai_results: Dict with XAI explanations
            heatmap_images: Dict with base64 encoded heatmap images (gradcam, scorecam)
            wsi_data: Dict with WSI analysis data
            survival_data: Dict with survival analysis data
        
        Returns:
            output_path: Path to generated PDF
        """
        from PyPDF2 import PdfMerger
        from io import BytesIO
        
        try:
            # Create in-memory PDF buffers
            part1_buffer = BytesIO()
            self._generate_part1_to_buffer(part1_buffer, patient_info, prediction_results, xai_results, heatmap_images)
            part1_buffer.seek(0)
            
            # Merge all parts
            merger = PdfMerger()
            merger.append(part1_buffer)
            
            # Part 2: WSI + Survival Analysis (if available)
            if wsi_data or survival_data:
                part2_buffer = BytesIO()
                self._generate_part2_to_buffer(part2_buffer, wsi_data, survival_data)
                part2_buffer.seek(0)
                merger.append(part2_buffer)
            
            # Write final merged PDF
            merger.write(output_path)
            merger.close()
            
            print(f"[PDF] Report generated successfully: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"[PDF] Error generating report: {e}")
            raise
    
    def _generate_part1_to_buffer(self, buffer, patient_info, prediction_results, xai_results, heatmap_images):
        """Generate Part 1: Patient Info + CNN + Tabular + XAI (IN-MEMORY)"""
        c = canvas.Canvas(buffer, pagesize=letter)
        
        # Header
        self._draw_header(c)
        
        # Patient Information
        y_pos = self._draw_patient_info(c, patient_info, start_y=self.page_height - 110)
        
        # Check if we need a new page
        if y_pos < 400:
            c.showPage()
            self._draw_header(c)
            y_pos = self.page_height - 110
        
        # Stage-1 CNN Results (DETAILED)
        y_pos = self._draw_cnn_results_detailed(c, prediction_results, start_y=y_pos - 35)
        
        # Check if we need a new page
        if y_pos < 350:
            c.showPage()
            self._draw_header(c)
            y_pos = self.page_height - 110
        
        # Stage-2 Tabular Results (DETAILED)
        if prediction_results.get('stage2_tabular'):
            y_pos = self._draw_tabular_results_detailed(c, prediction_results, start_y=y_pos - 35)
        
        # XAI Heatmaps - NEW PAGE
        if heatmap_images:
            c.showPage()
            self._draw_header(c)
            y_pos = self._draw_xai_heatmaps_detailed(c, heatmap_images, start_y=self.page_height - 110)
        
        # Risk Assessment
        if y_pos < 200:
            c.showPage()
            self._draw_header(c)
            y_pos = self.page_height - 110
        
        y_pos = self._draw_risk_assessment(c, xai_results, start_y=y_pos - 35)
        
        # Footer
        self._draw_footer(c)
        c.save()
    
    def _generate_part2_to_buffer(self, buffer, wsi_data, survival_data):
        """Generate Part 2: WSI + Survival Analysis (IN-MEMORY)"""
        c = canvas.Canvas(buffer, pagesize=letter)
        
        self._draw_header(c)
        y_pos = self.page_height - 110
        
        # WSI Analysis
        if wsi_data:
            y_pos = self._draw_wsi_analysis(c, wsi_data, start_y=y_pos)
            print(f"[PDF] WSI section drawn, y_pos: {y_pos}")
        else:
            print("[PDF] No WSI data to draw")
        
        # Survival Analysis - NEW PAGE
        if survival_data:
            c.showPage()
            self._draw_header(c)
            y_pos = self._draw_survival_analysis(c, survival_data, start_y=self.page_height - 110)
            print(f"[PDF] Survival section drawn, y_pos: {y_pos}")
        else:
            print("[PDF] No survival data to draw")
        
        # Footer
        self._draw_footer(c)
        c.save()
        print("[PDF] Part 2 generated in-memory")
    
    def _draw_cnn_results_detailed(self, c, prediction_results, start_y):
        """Draw detailed Stage-1 CNN results"""
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, start_y, "Stage-1: CNN Model")
        
        y = start_y - 20
        c.setFont("Helvetica", 11)
        c.drawString(50, y, "HetFusionNet v2 (Swin-ViT + CrossViT)")
        y -= 30
        
        # CNN Results
        cnn_data = prediction_results.get('stage1_cnn', {})
        
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Prediction:")
        c.setFont("Helvetica", 12)
        
        pred = cnn_data.get('prediction', 'N/A')
        if pred == 'OSCC':
            c.setFillColorRGB(0.8, 0.2, 0.2)
        else:
            c.setFillColorRGB(0.2, 0.6, 0.2)
        c.drawString(200, y, pred)
        c.setFillColorRGB(0, 0, 0)
        y -= 20
        
        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y, "Confidence:")
        c.setFont("Helvetica", 11)
        c.drawString(200, y, f"{cnn_data.get('confidence', 0):.2f}%")
        y -= 18
        
        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y, "Confidence Level:")
        c.setFont("Helvetica", 11)
        c.drawString(200, y, cnn_data.get('confidence_level', 'N/A'))
        y -= 18
        
        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y, "Uncertainty:")
        c.setFont("Helvetica", 11)
        c.drawString(200, y, f"{cnn_data.get('uncertainty', 0):.4f}")
        y -= 18
        
        # Preprocessing info
        preprocessing = cnn_data.get('preprocessing', {})
        if preprocessing:
            c.setFont("Helvetica-Bold", 11)
            c.drawString(50, y, "Preprocessing:")
            c.setFont("Helvetica", 10)
            
            blur_check = preprocessing.get('blur_check', 'N/A')
            tissue_check = preprocessing.get('tissue_check', 'N/A')
            macenko = 'Applied' if preprocessing.get('macenko_applied') else 'Not Applied'
            
            c.drawString(200, y, f"Blur Check: {blur_check}, Tissue Check: {tissue_check}, Macenko: {macenko}")
            y -= 20
        
        return y
    
    def _draw_tabular_results_detailed(self, c, prediction_results, start_y):
        """Draw detailed Stage-2 Tabular results"""
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, start_y, "Stage-2: Tabular Model")
        
        y = start_y - 20
        c.setFont("Helvetica", 11)
        c.drawString(50, y, "Clinical Data Analysis")
        y -= 30
        
        tabular = prediction_results.get('stage2_tabular', {})
        
        items = [
            ("Cancer Stage:", f"Stage {tabular.get('cancer_stage', 'N/A')}"),
            ("Stage Confidence:", f"{tabular.get('stage_confidence', 0):.0f}%"),
            ("5-Year Survival:", f"{tabular.get('survival_rate_5yr', 0):.0f}%"),
            ("Treatment:", tabular.get('treatment_type', 'N/A')),
            ("Estimated Cost:", f"${tabular.get('cost_usd', 0):,.0f}"),
            ("Economic Burden:", f"{tabular.get('economic_burden_days', 0)} days")
        ]
        
        c.setFont("Helvetica-Bold", 11)
        for label, value in items:
            c.drawString(50, y, label)
            c.setFont("Helvetica", 11)
            c.drawString(200, y, str(value))
            c.setFont("Helvetica-Bold", 11)
            y -= 18
        
        return y
    
    def _draw_xai_heatmaps_detailed(self, c, heatmap_images, start_y):
        """Draw detailed XAI heatmaps with descriptions - SIDE BY SIDE with proper spacing"""
        import base64
        from io import BytesIO
        from PIL import Image as PILImage
        
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, start_y, "Explainable AI (XAI) - Visual Explanations")
        
        y = start_y - 25
        c.setFont("Helvetica", 10)
        c.drawString(50, y, "Heatmaps showing which regions influenced the AI's decision")
        y -= 40
        
        img_width = 250
        img_height = 250
        left_x = 50
        right_x = 320
        
        # Draw both heatmaps side by side
        gradcam_available = 'gradcam' in heatmap_images and heatmap_images['gradcam']
        scorecam_available = 'scorecam' in heatmap_images and heatmap_images['scorecam']
        
        if gradcam_available or scorecam_available:
            # Grad-CAM++ (LEFT)
            if gradcam_available:
                try:
                    c.setFont("Helvetica-Bold", 13)
                    c.drawString(left_x, y, "Grad-CAM++ Heatmap")
                    y_desc = y - 18
                    c.setFont("Helvetica", 9)
                    c.drawString(left_x, y_desc, "Advanced gradient-based attention with sharp localization")
                    
                    gradcam_data = heatmap_images['gradcam']
                    if gradcam_data.startswith('data:image'):
                        gradcam_data = gradcam_data.split(',')[1]
                    
                    img_data = base64.b64decode(gradcam_data)
                    img = PILImage.open(BytesIO(img_data))
                    
                    y_img = y_desc - 30
                    c.drawImage(ImageReader(img), left_x, y_img - img_height, width=img_width, height=img_height)
                    
                    y_caption = y_img - img_height - 10
                    c.setFont("Helvetica", 8)
                    c.drawString(left_x, y_caption, "Red/Yellow regions: Areas with highest diagnostic importance")
                    c.drawString(left_x, y_caption - 10, "Processing: Adaptive thresholding + sharpening + contrast enhancement")
                    
                except Exception as e:
                    print(f"[PDF] Error drawing Grad-CAM++: {e}")
            
            # Layer-CAM (RIGHT)
            if scorecam_available:
                try:
                    c.setFont("Helvetica-Bold", 13)
                    c.drawString(right_x, y, "Layer-CAM Heatmap")
                    y_desc = y - 18
                    c.setFont("Helvetica", 9)
                    c.drawString(right_x, y_desc, "Alternative visualization for validation and comparison")
                    
                    scorecam_data = heatmap_images['scorecam']
                    if scorecam_data.startswith('data:image'):
                        scorecam_data = scorecam_data.split(',')[1]
                    
                    img_data = base64.b64decode(scorecam_data)
                    img = PILImage.open(BytesIO(img_data))
                    
                    y_img = y_desc - 30
                    c.drawImage(ImageReader(img), right_x, y_img - img_height, width=img_width, height=img_height)
                    
                    y_caption = y_img - img_height - 10
                    c.setFont("Helvetica", 8)
                    c.drawString(right_x, y_caption, "Colored regions: Complementary attention visualization")
                    c.drawString(right_x, y_caption - 10, "Purpose: Cross-validation of AI decision-making process")
                    
                except Exception as e:
                    print(f"[PDF] Error drawing Layer-CAM: {e}")
            
            y = y_desc - img_height - 50
        
        return y
    
    def _draw_header(self, c):
        """Draw report header with proper spacing"""
        c.setFont("Helvetica-Bold", 22)
        c.drawString(50, self.page_height - 50, "NeuroPlex Diagnostic Report")
        
        c.setFont("Helvetica", 10)
        c.drawString(50, self.page_height - 72, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Draw line
        c.setLineWidth(1.5)
        c.line(50, self.page_height - 85, self.page_width - 50, self.page_height - 85)
    
    def _draw_patient_info(self, c, patient_info, start_y):
        """Draw patient information section"""
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, start_y, "Patient Information")
        
        c.setFont("Helvetica", 11)
        y = start_y - 25
        
        # Get patient data from patient_info
        patient_data = patient_info.get('patient_data', {})
        
        info_items = [
            ("Patient ID:", patient_info.get('patient_id', 'N/A')),
            ("Patient Name:", patient_data.get('Patient_Name', 'N/A')),
            ("Age:", str(patient_data.get('Age', patient_info.get('age', 'N/A')))),
            ("Gender:", patient_data.get('Gender', patient_info.get('gender', 'N/A'))),
            ("Country:", patient_data.get('Country', 'N/A')),
            ("Hospital ID:", patient_info.get('hospital_id', 'N/A'))
        ]
        
        for label, value in info_items:
            c.drawString(50, y, f"{label}")
            c.drawString(200, y, str(value))
            y -= 20
        
        y -= 10
        
        # Lifestyle Factors
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Lifestyle Factors")
        y -= 18
        
        c.setFont("Helvetica", 10)
        lifestyle_items = [
            ("Tobacco Use:", patient_data.get('Tobacco Use', 'N/A')),
            ("Alcohol Consumption:", patient_data.get('Alcohol Consumption', 'N/A')),
            ("Betel Quid Use:", patient_data.get('Betel Quid Use', 'N/A')),
            ("Diet:", patient_data.get('Diet (Fruits & Vegetables Intake)', 'N/A'))
        ]
        
        for label, value in lifestyle_items:
            c.drawString(70, y, label)
            # Color code Yes/No
            if value == 'Yes':
                c.setFillColorRGB(0.8, 0.2, 0.2)  # Red
            elif value == 'No':
                c.setFillColorRGB(0.2, 0.6, 0.2)  # Green
            c.drawString(250, y, str(value))
            c.setFillColorRGB(0, 0, 0)  # Reset
            y -= 15
        
        y -= 10
        
        # Medical History
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Medical History")
        y -= 18
        
        c.setFont("Helvetica", 10)
        medical_items = [
            ("HPV Infection:", patient_data.get('HPV Infection', 'N/A')),
            ("Family History of Cancer:", patient_data.get('Family History of Cancer', 'N/A')),
            ("Compromised Immune System:", patient_data.get('Compromised Immune System', 'N/A')),
            ("Chronic Sun Exposure:", patient_data.get('Chronic Sun Exposure', 'N/A')),
            ("Poor Oral Hygiene:", patient_data.get('Poor Oral Hygiene', 'N/A'))
        ]
        
        for label, value in medical_items:
            c.drawString(70, y, label)
            # Color code Yes/No
            if value == 'Yes':
                c.setFillColorRGB(0.8, 0.2, 0.2)  # Red
            elif value == 'No':
                c.setFillColorRGB(0.2, 0.6, 0.2)  # Green
            c.drawString(300, y, str(value))
            c.setFillColorRGB(0, 0, 0)  # Reset
            y -= 15
        
        y -= 10
        
        # Symptoms
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Symptoms")
        y -= 18
        
        c.setFont("Helvetica", 10)
        symptom_items = [
            ("Oral Lesions:", patient_data.get('Oral Lesions', 'N/A')),
            ("Unexplained Bleeding:", patient_data.get('Unexplained Bleeding', 'N/A')),
            ("Difficulty Swallowing:", patient_data.get('Difficulty Swallowing', 'N/A')),
            ("White/Red Patches in Mouth:", patient_data.get('White or Red Patches in Mouth', 'N/A'))
        ]
        
        for label, value in symptom_items:
            c.drawString(70, y, label)
            # Color code Yes/No
            if value == 'Yes':
                c.setFillColorRGB(0.9, 0.5, 0.1)  # Orange
            elif value == 'No':
                c.setFillColorRGB(0.2, 0.6, 0.2)  # Green
            c.drawString(300, y, str(value))
            c.setFillColorRGB(0, 0, 0)  # Reset
            y -= 15
        
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
        """Draw risk assessment section with proper spacing"""
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, start_y, "Risk Stratification")
        
        y = start_y - 30
        
        risk_info = xai_results.get('risk', {})
        tier = risk_info.get('tier', 'UNKNOWN')
        action = risk_info.get('action', 'N/A')
        color_hex = risk_info.get('color', '#000000')
        
        # Convert hex to RGB
        color_rgb = tuple(int(color_hex[i:i+2], 16)/255 for i in (1, 3, 5))
        
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Risk Tier:")
        c.setFont("Helvetica-Bold", 14)
        c.setFillColorRGB(*color_rgb)
        c.drawString(200, y, tier)
        c.setFillColorRGB(0, 0, 0)
        
        y -= 35
        
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Recommended Action:")
        y -= 22
        
        # Wrap action text if too long
        c.setFont("Helvetica", 11)
        max_width = 500
        words = action.split()
        line = ""
        for word in words:
            test_line = line + word + " "
            if c.stringWidth(test_line, "Helvetica", 11) < max_width:
                line = test_line
            else:
                c.drawString(70, y, line)
                y -= 16
                line = word + " "
        if line:
            c.drawString(70, y, line)
            y -= 25
        
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
    
    def _draw_wsi_analysis(self, c, wsi_data, start_y):
        """Draw WSI Analysis section with detailed statistics and heatmap"""
        import base64
        from io import BytesIO
        from PIL import Image as PILImage
        
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, start_y, "Whole Slide Image (WSI) Analysis")
        
        y = start_y - 20
        c.setFont("Helvetica", 10)
        c.drawString(50, y, "Spatial heatmap showing cancer probability across tissue regions")
        y -= 35
        
        # WSI Overview
        c.setFont("Helvetica-Bold", 13)
        c.drawString(50, y, "WSI Overview")
        y -= 20
        
        dimensions = wsi_data.get('dimensions', [0, 0])
        total_tiles = wsi_data.get('total_tiles', 0)
        tissue_pct = wsi_data.get('tissue_percentage', 0) * 100
        
        c.setFont("Helvetica-Bold", 10)
        c.drawString(70, y, "Dimensions:")
        c.setFont("Helvetica", 10)
        c.drawString(200, y, f"{dimensions[0]} × {dimensions[1]}")
        y -= 15
        
        c.setFont("Helvetica-Bold", 10)
        c.drawString(70, y, "Tiles Analyzed:")
        c.setFont("Helvetica", 10)
        c.drawString(200, y, str(total_tiles))
        y -= 15
        
        c.setFont("Helvetica-Bold", 10)
        c.drawString(70, y, "Tissue Coverage:")
        c.setFont("Helvetica", 10)
        c.drawString(200, y, f"{tissue_pct:.1f}%")
        y -= 25
        
        # Spatial Statistics
        c.setFont("Helvetica-Bold", 13)
        c.drawString(50, y, "Spatial Statistics")
        y -= 20
        
        cancer_tiles = wsi_data.get('cancer_tiles', 0)
        normal_tiles = wsi_data.get('normal_tiles', 0)
        avg_conf = wsi_data.get('avg_confidence', 0) * 100
        
        c.setFont("Helvetica-Bold", 10)
        c.drawString(70, y, "Cancer Tiles:")
        c.setFont("Helvetica", 10)
        c.setFillColorRGB(0.8, 0.2, 0.2)
        c.drawString(200, y, str(cancer_tiles))
        c.setFillColorRGB(0, 0, 0)
        y -= 15
        
        c.setFont("Helvetica-Bold", 10)
        c.drawString(70, y, "Normal Tiles:")
        c.setFont("Helvetica", 10)
        c.setFillColorRGB(0.2, 0.6, 0.2)
        c.drawString(200, y, str(normal_tiles))
        c.setFillColorRGB(0, 0, 0)
        y -= 15
        
        c.setFont("Helvetica-Bold", 10)
        c.drawString(70, y, "Avg Confidence:")
        c.setFont("Helvetica", 10)
        c.drawString(200, y, f"{avg_conf:.1f}%")
        y -= 30
        
        # Spatial Heatmap
        c.setFont("Helvetica-Bold", 13)
        c.drawString(50, y, "Spatial Heatmap")
        y -= 15
        c.setFont("Helvetica", 9)
        c.drawString(50, y, "Red regions indicate higher cancer probability")
        y -= 25
        
        if 'heatmap_base64' in wsi_data and wsi_data['heatmap_base64']:
            try:
                heatmap_data = wsi_data['heatmap_base64']
                img_data = base64.b64decode(heatmap_data)
                img = PILImage.open(BytesIO(img_data))
                
                # Calculate size to fit
                max_width = 500
                max_height = 300
                img_width, img_height = img.size
                scale = min(max_width/img_width, max_height/img_height)
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                
                c.drawImage(ImageReader(img), 50, y - new_height, width=new_width, height=new_height)
                
                y -= new_height + 15
                
                c.setFont("Helvetica", 8)
                c.drawString(50, y, "Interpretation: Each tile represents a 224×224 region analyzed by the CNN model.")
                y -= 10
                c.drawString(50, y, "Color Scale: Blue (Normal) → Yellow → Red (Cancer)")
                y -= 15
                
            except Exception as e:
                print(f"[PDF] Error drawing WSI heatmap: {e}")
                c.setFont("Helvetica", 10)
                c.drawString(50, y, "WSI heatmap visualization unavailable")
                y -= 20
        
        return y
    
    def _draw_survival_analysis(self, c, survival_data, start_y):
        """Draw comprehensive Survival Analysis section"""
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, start_y, "Survival Analysis & Prognosis")
        
        y = start_y - 20
        c.setFont("Helvetica", 10)
        c.drawString(50, y, "Personalized survival predictions based on your cancer stage and risk factors")
        y -= 35
        
        curve = survival_data.get('survival_curve', {})
        milestones = curve.get('milestones', {})
        median_survival = curve.get('median_survival_months')
        
        # Key Survival Metrics - Display as cards
        c.setFont("Helvetica-Bold", 12)
        
        # 1-Year Survival
        c.drawString(50, y, "1-Year Survival")
        c.setFont("Helvetica-Bold", 20)
        c.setFillColorRGB(0.2, 0.6, 0.8)
        c.drawString(50, y - 25, f"{milestones.get('1_year', 0)*100:.1f}%")
        c.setFillColorRGB(0, 0, 0)
        
        # 3-Year Survival
        c.setFont("Helvetica-Bold", 12)
        c.drawString(180, y, "3-Year Survival")
        c.setFont("Helvetica-Bold", 20)
        c.setFillColorRGB(0.9, 0.6, 0.2)
        c.drawString(180, y - 25, f"{milestones.get('3_year', 0)*100:.1f}%")
        c.setFillColorRGB(0, 0, 0)
        
        # 5-Year Survival
        c.setFont("Helvetica-Bold", 12)
        c.drawString(310, y, "5-Year Survival")
        c.setFont("Helvetica-Bold", 20)
        c.setFillColorRGB(0.8, 0.3, 0.3)
        c.drawString(310, y - 25, f"{milestones.get('5_year', 0)*100:.1f}%")
        c.setFillColorRGB(0, 0, 0)
        
        # Median Survival
        c.setFont("Helvetica-Bold", 12)
        c.drawString(440, y, "Median Survival")
        c.setFont("Helvetica-Bold", 16)
        if median_survival:
            years = median_survival // 12
            months = median_survival % 12
            c.drawString(440, y - 25, f"{years}y {months}m")
        else:
            c.drawString(440, y - 25, ">10y")
        
        y -= 55
        
        # Kaplan-Meier Curve
        c.setFont("Helvetica-Bold", 13)
        c.drawString(50, y, "Kaplan-Meier Survival Curve")
        y -= 15
        c.setFont("Helvetica", 9)
        c.drawString(50, y, "Blue line: Your personalized survival probability")
        y -= 10
        c.drawString(50, y, "Shaded area: 95% confidence interval")
        y -= 10
        c.drawString(50, y, "Interpretation: Higher curve = better prognosis")
        y -= 25
        
        # Debug: Check if curve_base64 exists
        print(f"[PDF-SURVIVAL] Checking for curve_base64 in survival data...")
        print(f"[PDF-SURVIVAL] curve keys: {list(curve.keys())}")
        print(f"[PDF-SURVIVAL] Has curve_base64: {'curve_base64' in curve}")
        
        # Draw Kaplan-Meier curve image
        if 'curve_base64' in curve and curve['curve_base64']:
            try:
                import base64
                from io import BytesIO
                from PIL import Image as PILImage
                
                print(f"[PDF-SURVIVAL] Found curve_base64, size: {len(curve['curve_base64'])} bytes")
                
                curve_data = curve['curve_base64']
                img_data = base64.b64decode(curve_data)
                img = PILImage.open(BytesIO(img_data))
                
                print(f"[PDF-SURVIVAL] Decoded image, size: {img.size}")
                
                # Calculate size to fit
                max_width = 500
                max_height = 280
                img_width, img_height = img.size
                scale = min(max_width/img_width, max_height/img_height)
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                
                print(f"[PDF-SURVIVAL] Drawing curve at y={y}, size={new_width}x{new_height}")
                
                c.drawImage(ImageReader(img), 50, y - new_height, width=new_width, height=new_height)
                
                y -= new_height + 15
                
                print(f"[PDF-SURVIVAL] ✓ Kaplan-Meier curve drawn successfully")
                
            except Exception as e:
                print(f"[PDF-SURVIVAL] ERROR drawing Kaplan-Meier curve: {e}")
                import traceback
                traceback.print_exc()
                c.setFont("Helvetica", 10)
                c.drawString(50, y, "Kaplan-Meier curve visualization error")
                y -= 20
        else:
            print(f"[PDF-SURVIVAL] WARNING: curve_base64 not found in survival data")
            c.setFont("Helvetica", 10)
            c.drawString(50, y, "Kaplan-Meier curve visualization unavailable")
            y -= 20
        
        y -= 10
        
        # Population Comparison
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, "Comparison with Population Average")
        y -= 25
        
        comparison = survival_data.get('population_comparison', {})
        comp_data = comparison.get('comparison', {})
        
        if comp_data:
            # Table header
            c.setFont("Helvetica-Bold", 10)
            c.drawString(50, y, "Timepoint")
            c.drawString(150, y, "You")
            c.drawString(230, y, "Average")
            c.drawString(330, y, "Difference")
            y -= 15
            
            c.setFont("Helvetica", 9)
            for timepoint, label in [('1_year', '1 Year'), ('3_year', '3 Years'), ('5_year', '5 Years'), ('10_year', '10 Years')]:
                if timepoint in comp_data:
                    data = comp_data[timepoint]
                    patient_val = data.get('patient', 0) * 100
                    pop_val = data.get('population', 0) * 100
                    diff = data.get('difference', 0) * 100
                    
                    c.drawString(50, y, label)
                    c.drawString(150, y, f"{patient_val:.1f}%")
                    c.drawString(230, y, f"{pop_val:.1f}%")
                    
                    # Color code difference
                    if diff < 0:
                        c.setFillColorRGB(0.8, 0.2, 0.2)
                        c.drawString(330, y, f"{diff:.1f}%")
                    else:
                        c.setFillColorRGB(0.2, 0.6, 0.2)
                        c.drawString(330, y, f"+{diff:.1f}%")
                    c.setFillColorRGB(0, 0, 0)
                    
                    y -= 12
            
            y -= 15
            
            # Assessment
            assessment = comparison.get('assessment', 'N/A')
            avg_diff = comparison.get('avg_difference_percent', 0)
            
            c.setFont("Helvetica-Bold", 11)
            if assessment == 'Better than average':
                c.setFillColorRGB(0.2, 0.6, 0.2)
                c.drawString(50, y, f"✓ {assessment} (+{avg_diff:.1f}% vs population)")
            elif assessment == 'Below average':
                c.setFillColorRGB(0.8, 0.2, 0.2)
                c.drawString(50, y, f"⚠ {assessment} ({avg_diff:.1f}% vs population)")
            else:
                c.setFillColorRGB(0.2, 0.4, 0.8)
                c.drawString(50, y, f"ℹ {assessment}")
            c.setFillColorRGB(0, 0, 0)
            y -= 30
        
        # Check if we need a new page for recommendations
        if y < 300:
            c.showPage()
            self._draw_header(c)
            y = self.page_height - 100
        
        # Personalized Recommendations
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, "Personalized Recommendations")
        y -= 25
        
        recommendations = survival_data.get('recommendations', [])
        
        if recommendations:
            c.setFont("Helvetica", 9)
            for i, rec in enumerate(recommendations[:6], 1):
                priority = rec.get('priority', 'MEDIUM')
                category = rec.get('category', 'General')
                recommendation = rec.get('recommendation', '')
                impact = rec.get('impact', '')
                
                # Priority badge
                if priority == 'HIGH':
                    priority_symbol = "⚠"
                    c.setFillColorRGB(0.8, 0.2, 0.2)
                elif priority == 'MEDIUM':
                    priority_symbol = "⚡"
                    c.setFillColorRGB(0.9, 0.5, 0.1)
                else:
                    priority_symbol = "ℹ"
                    c.setFillColorRGB(0.2, 0.4, 0.8)
                
                c.setFont("Helvetica-Bold", 10)
                c.drawString(50, y, f"{priority_symbol} {priority}")
                c.setFillColorRGB(0, 0, 0)
                
                c.setFont("Helvetica-Bold", 9)
                c.drawString(120, y, category)
                y -= 12
                
                # Recommendation text
                c.setFont("Helvetica", 9)
                max_width = 480
                words = recommendation.split()
                line = ""
                for word in words:
                    test_line = line + word + " "
                    if c.stringWidth(test_line, "Helvetica", 9) < max_width:
                        line = test_line
                    else:
                        c.drawString(70, y, line)
                        y -= 10
                        line = word + " "
                if line:
                    c.drawString(70, y, line)
                    y -= 10
                
                # Impact
                c.setFont("Helvetica-Oblique", 8)
                c.drawString(70, y, f"Impact: {impact}")
                y -= 18
                
                # Check if we need a new page
                if y < 150 and i < len(recommendations):
                    c.showPage()
                    self._draw_header(c)
                    y = self.page_height - 100
                    c.setFont("Helvetica-Bold", 12)
                    c.drawString(50, y, "Personalized Recommendations (continued)")
                    y -= 20
                    c.setFont("Helvetica", 9)
        
        return y
    
    def _draw_model_info(self, c, xai_results):
        """Draw model information at bottom"""
        y = 100
        
        c.setFont("Helvetica-Bold", 10)
        c.drawString(50, y, "Model Information")
        
        c.setFont("Helvetica", 9)
        y -= 15
        
        metadata = xai_results.get('metadata', {})
        model_name = metadata.get('model', 'NeuroPlex v2.0')
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
        c.drawString(50, 20, "© 2024 NeuroPlex - AI-Powered Oral Cancer Detection System")


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
                "reference": "Organization/neuroplex",
                "display": "NeuroPlex AI System"
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
        
        # FETCH PATIENT DATA FROM MONGODB IF NOT PROVIDED
        patient_data = complete_result.get('patient_data', {})
        
        # If patient_data is empty, try to fetch from MongoDB
        if not patient_data or all(v in ['N/A', '', None] for v in patient_data.values()):
            try:
                from pymongo import MongoClient
                import os
                from dotenv import load_dotenv
                
                load_dotenv()
                MONGODB_URI = os.getenv('MONGODB_URI')
                DATABASE_NAME = os.getenv('DATABASE_NAME', 'neuroplex')
                
                if MONGODB_URI:
                    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
                    db = client[DATABASE_NAME]
                    predictions_collection = db['predictions']
                    
                    # Find the most recent prediction for this patient
                    prediction = predictions_collection.find_one(
                        {'patient_id': patient_id},
                        sort=[('timestamp', -1)]
                    )
                    
                    if prediction and prediction.get('patient_data'):
                        patient_data = prediction['patient_data']
                        print(f"[PDF] Fetched patient data from MongoDB for {patient_id}")
                    else:
                        print(f"[PDF] No patient data found in MongoDB for {patient_id}")
                else:
                    print("[PDF] MongoDB URI not configured")
                    
            except Exception as e:
                print(f"[PDF] Failed to fetch patient data from MongoDB: {e}")
        
        # Prepare patient info with full patient data
        patient_info = {
            'patient_id': patient_id,
            'age': patient_data.get('Age', 'N/A'),
            'gender': patient_data.get('Gender', 'N/A'),
            'hospital_id': complete_result.get('hospital_id', 'N/A'),
            'patient_data': patient_data  # Include full patient data
        }
        
        # Prepare prediction results
        stage1 = complete_result.get('stage1_cnn', {})
        stage2 = complete_result.get('stage2_tabular', {})
        
        prediction_results = {
            'class': 1 if complete_result.get('final_prediction') == 'OSCC' else 0,
            'confidence': complete_result.get('final_confidence', 0) / 100,
            'uncertainty': stage1.get('uncertainty', 0),
            'stage': stage2.get('cancer_stage', 'N/A') if stage2 else 'N/A',
            'stage1_cnn': stage1,
            'stage2_tabular': stage2
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
                'model': 'NeuroPlex v2.0',
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
        
        # Extract WSI data
        wsi_data = complete_result.get('wsi_result')
        
        # Extract Survival data
        survival_data = complete_result.get('survival_analysis')
        
        # Generate PDF
        pdf_path = os.path.join(output_dir, f'report_{patient_id}_{timestamp}.pdf')
        
        print(f"[PDF] Generating report with:")
        print(f"[PDF] - Patient data: {bool(patient_data)}")
        print(f"[PDF] - XAI heatmaps: {bool(heatmap_images)}")
        print(f"[PDF] - WSI data: {bool(wsi_data)}")
        print(f"[PDF] - Survival data: {bool(survival_data)}")
        
        self.pdf_generator.generate_report(pdf_path, patient_info, prediction_results, xai_results, heatmap_images, wsi_data, survival_data)
        
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
                'model': 'NeuroPlex v2.0',
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
    print("NeuroPlex Clinical Report Generator")
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
