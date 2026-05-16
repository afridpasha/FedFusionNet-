"""
Macenko Stain Normalization for Histopathology Images
Preprocessing pipeline for oral cancer detection
"""

import numpy as np
import cv2
from PIL import Image


class MacenkoNormalizer:
    """
    Macenko Stain Normalization
    Reference: "A method for normalizing histology slides for quantitative analysis"
    """
    
    def __init__(self):
        # Reference stain matrix (H&E)
        self.target_stains = np.array([
            [0.5626, 0.2159],
            [0.7201, 0.8012],
            [0.4062, 0.5581]
        ])
        self.target_concentrations = np.array([
            [1.9705, 1.0308]
        ])
        
    def normalize(self, image):
        """
        Apply Macenko normalization to image
        
        Args:
            image: PIL Image or numpy array (H×W×3, RGB)
        
        Returns:
            normalized_image: numpy array (H×W×3, RGB)
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure RGB
        if image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]
        
        # Reshape to 2D
        h, w, c = image.shape
        image_2d = image.reshape(-1, 3)
        
        # Convert RGB to Optical Density (OD)
        od = self._rgb_to_od(image_2d)
        
        # Remove transparent pixels
        od_hat = od[~np.any(od < 0.15, axis=1)]
        
        if od_hat.shape[0] < 2:
            # Not enough tissue, return original
            return image
        
        # Compute eigenvectors (stain directions)
        eigvals, eigvecs = np.linalg.eigh(np.cov(od_hat.T))
        eigvecs = eigvecs[:, [1, 2]]  # Take top 2
        
        # Project on plane
        that = od_hat.dot(eigvecs)
        phi = np.arctan2(that[:, 1], that[:, 0])
        
        # Find robust extremes (1st and 99th percentile)
        min_phi = np.percentile(phi, 1)
        max_phi = np.percentile(phi, 99)
        
        v1 = eigvecs.dot(np.array([np.cos(min_phi), np.sin(min_phi)]))
        v2 = eigvecs.dot(np.array([np.cos(max_phi), np.sin(max_phi)]))
        
        # Normalize stain matrix
        if v1[0] > v2[0]:
            he = np.array([v1, v2]).T
        else:
            he = np.array([v2, v1]).T
        
        # Normalize columns
        he = he / np.linalg.norm(he, axis=0)
        
        # Calculate concentrations
        c = np.linalg.lstsq(he, od.T, rcond=None)[0].T
        
        # Normalize concentrations
        max_c = np.percentile(c, 99, axis=0)
        c = c / max_c * self.target_concentrations
        
        # Reconstruct image with target stains
        od_normalized = c.dot(self.target_stains.T)
        image_normalized = self._od_to_rgb(od_normalized)
        
        # Reshape back
        image_normalized = image_normalized.reshape(h, w, 3)
        image_normalized = np.clip(image_normalized, 0, 255).astype(np.uint8)
        
        return image_normalized
    
    def _rgb_to_od(self, rgb):
        """Convert RGB to Optical Density"""
        rgb = rgb.astype(np.float64)
        rgb = np.maximum(rgb, 1)  # Avoid log(0)
        od = -np.log10(rgb / 255.0)
        return od
    
    def _od_to_rgb(self, od):
        """Convert Optical Density to RGB"""
        rgb = 255 * np.exp(-od * np.log(10))
        return rgb


class QualityControl:
    """Quality control for histopathology images"""
    
    @staticmethod
    def check_blur(image, threshold=100):
        """
        Check if image is blurry using Laplacian variance
        
        Args:
            image: numpy array or PIL Image
            threshold: variance threshold (default 100)
        
        Returns:
            is_good: True if image is sharp enough
            variance: Laplacian variance value
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        return variance >= threshold, variance
    
    @staticmethod
    def check_tissue_content(image, threshold=0.1):
        """
        Check if image has enough tissue content
        
        Args:
            image: numpy array or PIL Image
            threshold: minimum tissue ratio (default 0.1 = 10%)
        
        Returns:
            is_good: True if enough tissue
            tissue_ratio: percentage of tissue pixels
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Otsu thresholding to separate tissue from background
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Calculate tissue ratio
        tissue_pixels = np.sum(binary < 128)  # Dark pixels = tissue
        total_pixels = binary.size
        tissue_ratio = tissue_pixels / total_pixels
        
        return tissue_ratio >= threshold, tissue_ratio


def preprocess_image_for_prediction(image, apply_quality_control=True):
    """
    Complete preprocessing pipeline for oral cancer detection
    
    Pipeline:
    1. Quality Control (optional)
       - Blur detection (Laplacian variance)
       - Tissue content check
    2. Macenko Stain Normalization
       - RGB → Optical Density
       - SVD decomposition
       - Normalize to reference H&E
    3. Resize to 240×240
    4. Normalize for model input
    
    Args:
        image: PIL Image or numpy array
        apply_quality_control: Whether to check quality (default True)
    
    Returns:
        preprocessed_tensor: torch.Tensor ready for model
        quality_info: dict with quality metrics
    """
    import torch
    from torchvision import transforms
    
    # Convert to PIL if numpy
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    quality_info = {
        'blur_check': 'skipped',
        'tissue_check': 'skipped',
        'macenko_applied': False
    }
    
    # ═══════════════════════════════════════════════════════
    # STEP 1: QUALITY CONTROL
    # ═══════════════════════════════════════════════════════
    if apply_quality_control:
        qc = QualityControl()
        
        # Check blur
        is_sharp, blur_variance = qc.check_blur(image)
        quality_info['blur_check'] = 'pass' if is_sharp else 'fail'
        quality_info['blur_variance'] = blur_variance
        
        if not is_sharp:
            print(f"⚠ Warning: Image may be blurry (variance={blur_variance:.2f})")
        
        # Check tissue content
        has_tissue, tissue_ratio = qc.check_tissue_content(image)
        quality_info['tissue_check'] = 'pass' if has_tissue else 'fail'
        quality_info['tissue_ratio'] = tissue_ratio
        
        if not has_tissue:
            print(f"⚠ Warning: Low tissue content ({tissue_ratio:.2%})")
    
    # ═══════════════════════════════════════════════════════
    # STEP 2: MACENKO STAIN NORMALIZATION
    # ═══════════════════════════════════════════════════════
    try:
        normalizer = MacenkoNormalizer()
        image_normalized = normalizer.normalize(image)
        image = Image.fromarray(image_normalized)
        quality_info['macenko_applied'] = True
    except Exception as e:
        print(f"⚠ Macenko normalization failed: {e}")
        print("  Using original image")
        quality_info['macenko_applied'] = False
    
    # ═══════════════════════════════════════════════════════
    # STEP 3: RESIZE & NORMALIZE FOR MODEL
    # ═══════════════════════════════════════════════════════
    # Using 240×240 to match checkpoint training (both Swin and CrossViT)
    transform = transforms.Compose([
        transforms.Resize((240, 240)),  # Match checkpoint training size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(image).unsqueeze(0)
    
    return tensor, quality_info
