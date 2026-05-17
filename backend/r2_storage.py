"""
Cloudflare R2 Storage Service for PDF Reports
Stores large PDF files (26+ MB) in R2 instead of MongoDB
"""

import boto3
from botocore.exceptions import ClientError
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class R2StorageService:
    """Cloudflare R2 storage for PDF reports"""
    
    def __init__(self):
        self.access_key = os.getenv('R2_ACCESS_KEY_ID')
        self.secret_key = os.getenv('R2_SECRET_ACCESS_KEY')
        self.endpoint_url = os.getenv('R2_ENDPOINT_URL')
        self.bucket_name = os.getenv('R2_BUCKET_NAME', 'neuroplex-reports')
        self.public_url = os.getenv('R2_PUBLIC_URL')  # Optional: https://pub-xxx.r2.dev/neuroplex-reports
        
        self.client = None
        self.available = False
        
        if self.access_key and self.secret_key and self.endpoint_url:
            try:
                self.client = boto3.client(
                    's3',
                    endpoint_url=self.endpoint_url,
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key,
                    region_name='auto'
                )
                # Test connection
                self.client.head_bucket(Bucket=self.bucket_name)
                self.available = True
                print(f"[R2] Connected to bucket: {self.bucket_name}")
            except Exception as e:
                print(f"[R2] Connection failed: {e}")
                self.available = False
        else:
            print("[R2] Credentials not configured in .env")
    
    def is_available(self):
        """Check if R2 storage is available"""
        return self.available
    
    def upload_pdf(self, pdf_path, patient_id):
        """
        Upload PDF to R2 storage
        
        Args:
            pdf_path: Local path to PDF file
            patient_id: Patient ID for naming
            
        Returns:
            dict: {'success': bool, 'url': str, 'key': str, 'size_mb': float}
        """
        if not self.available:
            return {'success': False, 'error': 'R2 storage not available'}
        
        try:
            # Generate R2 key (path in bucket)
            filename = f"clinical_report_{patient_id}.pdf"
            r2_key = f"reports/{patient_id}/{filename}"
            
            # Get file size
            file_size = os.path.getsize(pdf_path)
            size_mb = file_size / 1024 / 1024
            
            print(f"[R2] Uploading {filename} ({size_mb:.2f}MB) to {r2_key}...")
            
            # Upload to R2
            with open(pdf_path, 'rb') as f:
                self.client.put_object(
                    Bucket=self.bucket_name,
                    Key=r2_key,
                    Body=f,
                    ContentType='application/pdf',
                    Metadata={
                        'patient_id': patient_id,
                        'filename': filename
                    }
                )
            
            # Generate URL
            if self.public_url:
                # Public URL (if bucket has public access)
                url = f"{self.public_url}/{r2_key}"
            else:
                # Presigned URL (valid for 7 days)
                url = self.client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': self.bucket_name, 'Key': r2_key},
                    ExpiresIn=604800  # 7 days
                )
            
            print(f"[R2] ✓ Upload successful: {r2_key}")
            
            return {
                'success': True,
                'url': url,
                'key': r2_key,
                'size_mb': round(size_mb, 2),
                'bucket': self.bucket_name
            }
            
        except ClientError as e:
            print(f"[R2] Upload failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def download_pdf(self, r2_key, local_path=None):
        """
        Download PDF from R2 storage
        
        Args:
            r2_key: R2 object key
            local_path: Optional local path to save (if None, returns binary)
            
        Returns:
            bytes or str: PDF binary data or local path
        """
        if not self.available:
            return None
        
        try:
            print(f"[R2] Downloading {r2_key}...")
            
            response = self.client.get_object(
                Bucket=self.bucket_name,
                Key=r2_key
            )
            
            pdf_binary = response['Body'].read()
            
            if local_path:
                # Save to local file
                with open(local_path, 'wb') as f:
                    f.write(pdf_binary)
                print(f"[R2] ✓ Downloaded to {local_path}")
                return local_path
            else:
                # Return binary data
                print(f"[R2] ✓ Downloaded {len(pdf_binary)} bytes")
                return pdf_binary
                
        except ClientError as e:
            print(f"[R2] Download failed: {e}")
            return None
    
    def delete_pdf(self, r2_key):
        """Delete PDF from R2 storage"""
        if not self.available:
            return False
        
        try:
            print(f"[R2] Deleting {r2_key}...")
            
            self.client.delete_object(
                Bucket=self.bucket_name,
                Key=r2_key
            )
            
            print(f"[R2] ✓ Deleted {r2_key}")
            return True
            
        except ClientError as e:
            print(f"[R2] Delete failed: {e}")
            return False
    
    def get_presigned_url(self, r2_key, expires_in=3600):
        """
        Generate presigned URL for temporary access
        
        Args:
            r2_key: R2 object key
            expires_in: URL validity in seconds (default: 1 hour)
            
        Returns:
            str: Presigned URL
        """
        if not self.available:
            return None
        
        try:
            url = self.client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': r2_key},
                ExpiresIn=expires_in
            )
            return url
        except ClientError as e:
            print(f"[R2] Failed to generate presigned URL: {e}")
            return None

def create_r2_service():
    """Factory function to create R2 storage service"""
    return R2StorageService()
