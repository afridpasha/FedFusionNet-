"""
Email Service Module
Sends clinical reports to hospital email after prediction completion
"""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from datetime import datetime
import os


class EmailService:
    """
    Email service for sending clinical reports
    """
    
    def __init__(self, email_user, email_pass):
        """
        Initialize email service
        
        Args:
            email_user: Gmail address
            email_pass: Gmail app password
        """
        self.email_user = email_user
        self.email_pass = email_pass
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
    
    def send_clinical_report(self, recipient_email, hospital_name, patient_id, 
                            prediction_result, pdf_path=None, json_path=None):
        """
        Send clinical report email with PDF and JSON attachments
        
        Args:
            recipient_email: Hospital email address
            hospital_name: Name of the hospital
            patient_id: Patient ID
            prediction_result: Dictionary with prediction results
            pdf_path: Path to PDF report (optional)
            json_path: Path to JSON report (optional)
        
        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = recipient_email
            msg['Subject'] = f"FedFusionNet++ Clinical Report - Patient {patient_id}"
            
            # Create email body
            body = self._create_email_body(hospital_name, patient_id, prediction_result)
            msg.attach(MIMEText(body, 'html'))
            
            # Attach PDF if provided
            if pdf_path and os.path.exists(pdf_path):
                with open(pdf_path, 'rb') as f:
                    pdf_attachment = MIMEApplication(f.read(), _subtype='pdf')
                    pdf_attachment.add_header('Content-Disposition', 'attachment', 
                                            filename=f'clinical_report_{patient_id}.pdf')
                    msg.attach(pdf_attachment)
                print(f"[EMAIL] PDF attached: {pdf_path}")
            
            # Attach JSON if provided
            if json_path and os.path.exists(json_path):
                with open(json_path, 'rb') as f:
                    json_attachment = MIMEApplication(f.read(), _subtype='json')
                    json_attachment.add_header('Content-Disposition', 'attachment', 
                                             filename=f'clinical_report_{patient_id}.json')
                    msg.attach(json_attachment)
                print(f"[EMAIL] JSON attached: {json_path}")
            
            # Send email
            print(f"[EMAIL] Connecting to SMTP server...")
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            
            print(f"[EMAIL] Logging in...")
            server.login(self.email_user, self.email_pass)
            
            print(f"[EMAIL] Sending email to {recipient_email}...")
            server.send_message(msg)
            server.quit()
            
            print(f"[EMAIL] Email sent successfully to {recipient_email}")
            return True, "Email sent successfully"
            
        except smtplib.SMTPAuthenticationError:
            error_msg = "Email authentication failed. Check email credentials."
            print(f"[EMAIL ERROR] {error_msg}")
            return False, error_msg
            
        except smtplib.SMTPException as e:
            error_msg = f"SMTP error: {str(e)}"
            print(f"[EMAIL ERROR] {error_msg}")
            return False, error_msg
            
        except Exception as e:
            error_msg = f"Failed to send email: {str(e)}"
            print(f"[EMAIL ERROR] {error_msg}")
            return False, error_msg
    
    def _create_email_body(self, hospital_name, patient_id, prediction_result):
        """
        Create HTML email body with prediction results
        
        Args:
            hospital_name: Name of the hospital
            patient_id: Patient ID
            prediction_result: Dictionary with prediction results
        
        Returns:
            str: HTML email body
        """
        # Extract results
        final_prediction = prediction_result.get('final_prediction', 'N/A')
        final_confidence = prediction_result.get('final_confidence', 0)
        risk_level = prediction_result.get('risk_level', 'N/A')
        
        stage1 = prediction_result.get('stage1_cnn', {})
        cnn_prediction = stage1.get('prediction', 'N/A')
        cnn_confidence = stage1.get('confidence', 0)
        
        stage2 = prediction_result.get('stage2_tabular', {})
        cancer_stage = stage2.get('cancer_stage', 'N/A') if stage2 else 'N/A'
        survival_rate = stage2.get('survival_rate_5yr', 'N/A') if stage2 else 'N/A'
        
        # Determine color based on prediction
        if final_prediction == 'OSCC':
            prediction_color = '#dc3545'  # Red
            prediction_icon = '⚠️'
        else:
            prediction_color = '#28a745'  # Green
            prediction_icon = '✓'
        
        # Create HTML body
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    text-align: center;
                    border-radius: 10px 10px 0 0;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 24px;
                }}
                .content {{
                    background: #f8f9fa;
                    padding: 30px;
                    border-radius: 0 0 10px 10px;
                }}
                .prediction-box {{
                    background: white;
                    border-left: 5px solid {prediction_color};
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .prediction-result {{
                    font-size: 28px;
                    font-weight: bold;
                    color: {prediction_color};
                    margin: 10px 0;
                }}
                .info-table {{
                    width: 100%;
                    background: white;
                    border-radius: 5px;
                    overflow: hidden;
                    margin: 20px 0;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .info-table tr {{
                    border-bottom: 1px solid #e9ecef;
                }}
                .info-table tr:last-child {{
                    border-bottom: none;
                }}
                .info-table td {{
                    padding: 12px 15px;
                }}
                .info-table td:first-child {{
                    font-weight: bold;
                    color: #666;
                    width: 40%;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 2px solid #e9ecef;
                    color: #666;
                    font-size: 12px;
                }}
                .warning {{
                    background: #fff3cd;
                    border-left: 4px solid #ffc107;
                    padding: 15px;
                    margin: 20px 0;
                    border-radius: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🏥 FedFusionNet++ Clinical Report</h1>
                <p>AI-Powered Oral Cancer Detection System</p>
            </div>
            
            <div class="content">
                <p><strong>Dear {hospital_name},</strong></p>
                
                <p>This is an automated clinical report generated by the FedFusionNet++ AI system for patient <strong>{patient_id}</strong>.</p>
                
                <div class="prediction-box">
                    <div style="font-size: 14px; color: #666; margin-bottom: 10px;">FINAL DIAGNOSIS</div>
                    <div class="prediction-result">{prediction_icon} {final_prediction}</div>
                    <div style="margin-top: 10px;">
                        <strong>Confidence:</strong> {final_confidence}%<br>
                        <strong>Risk Level:</strong> {risk_level}
                    </div>
                </div>
                
                <h3 style="color: #667eea; margin-top: 30px;">📊 Detailed Results</h3>
                
                <table class="info-table">
                    <tr>
                        <td>Stage-1 CNN Prediction</td>
                        <td><strong>{cnn_prediction}</strong></td>
                    </tr>
                    <tr>
                        <td>CNN Confidence</td>
                        <td>{cnn_confidence}%</td>
                    </tr>
                    <tr>
                        <td>Cancer Stage</td>
                        <td><strong>Stage {cancer_stage}</strong></td>
                    </tr>
                    <tr>
                        <td>5-Year Survival Rate</td>
                        <td>{survival_rate}%</td>
                    </tr>
                    <tr>
                        <td>Report Generated</td>
                        <td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td>
                    </tr>
                </table>
                
                <div class="warning">
                    <strong>⚠️ Important Notice:</strong><br>
                    This report is generated by an AI system and should be used as a diagnostic aid only. 
                    Final diagnosis must be confirmed by a qualified pathologist through clinical examination 
                    and additional tests as necessary.
                </div>
                
                <p style="margin-top: 30px;">
                    <strong>Attachments:</strong><br>
                    • Clinical Report (PDF) - Complete detailed report<br>
                    • Clinical Report (JSON) - Machine-readable format
                </p>
                
                <p>For any questions or concerns, please contact your system administrator.</p>
                
                <div class="footer">
                    <p>
                        <strong>FedFusionNet++</strong><br>
                        Advanced AI System for Oral Cancer Detection<br>
                        Two-Stage Deep Learning Framework<br>
                        <br>
                        This is an automated email. Please do not reply.
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_body
    
    def send_test_email(self, recipient_email):
        """
        Send a test email to verify configuration
        
        Args:
            recipient_email: Email address to send test to
        
        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = recipient_email
            msg['Subject'] = "FedFusionNet++ - Email Configuration Test"
            
            body = """
            <html>
            <body style="font-family: Arial, sans-serif; padding: 20px;">
                <h2 style="color: #667eea;">Email Configuration Test</h2>
                <p>This is a test email from FedFusionNet++ system.</p>
                <p>If you received this email, your email configuration is working correctly!</p>
                <p style="margin-top: 30px; color: #666; font-size: 12px;">
                    Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_user, self.email_pass)
            server.send_message(msg)
            server.quit()
            
            return True, "Test email sent successfully"
            
        except Exception as e:
            return False, f"Failed to send test email: {str(e)}"


def create_email_service():
    """
    Create email service instance from environment variables
    
    Returns:
        EmailService: Configured email service instance or None if not configured
    """
    email_user = os.getenv('EMAIL_USER')
    email_pass = os.getenv('EMAIL_PASS')
    
    if not email_user or not email_pass:
        print("[EMAIL] Email credentials not configured in .env file")
        return None
    
    return EmailService(email_user, email_pass)
