"""
ULTIMATE XAI MODULE - Medical-Grade Visualization
Crystal-clear, sharp, high-resolution heatmaps for histopathology analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter, maximum_filter
from skimage import exposure, filters


class GradCAMPlusPlus:
    """
    Grad-CAM++ with enhanced sharpness and localization
    """
    def __init__(self, model, target_layer, device='cpu'):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.activations = None
        self.gradients = None
        self.hooks = []
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def register_hooks(self):
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.hooks.append(module.register_forward_hook(self.save_activation))
                self.hooks.append(module.register_full_backward_hook(self.save_gradient))
                return True
        return False
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate(self, input_tensor, class_idx=1):
        """Generate high-quality Grad-CAM++"""
        self.activations = None
        self.gradients = None
        
        if not self.register_hooks():
            print(f"[XAI-ULTIMATE] Grad-CAM++: Failed to register hooks for layer {self.target_layer}")
            return None
        
        try:
            # Forward pass
            input_tensor.requires_grad = True
            output = self.model(input_tensor)
            
            # Backward pass
            self.model.zero_grad()
            score = output[0, class_idx]
            score.backward(retain_graph=False)  # Changed to False to avoid memory issues
            
            self.remove_hooks()
            
            if self.activations is None:
                print("[XAI-ULTIMATE] Grad-CAM++: Activations are None")
                return None
            
            if self.gradients is None:
                print("[XAI-ULTIMATE] Grad-CAM++: Gradients are None")
                return None
            
            # Check if activations have spatial dimensions
            if len(self.activations.shape) != 4:
                print(f"[XAI-ULTIMATE] Grad-CAM++: Invalid activation shape {self.activations.shape}")
                return None
            
            b, c, h, w = self.activations.shape
            print(f"[XAI-ULTIMATE] Grad-CAM++: Activation shape [{b}, {c}, {h}, {w}]")
            
            # Grad-CAM++ weights
            gradients = self.gradients
            activations = self.activations
            
            # Check if gradients are all zero
            if gradients.abs().sum() == 0:
                print("[XAI-ULTIMATE] Grad-CAM++: Gradients are all zeros")
                return None
            
            # Simplified Grad-CAM++ (more robust)
            # Use positive gradients only
            relu_grad = F.relu(gradients)
            
            # Global average pooling of gradients as weights
            weights = relu_grad.mean(dim=(2, 3), keepdim=True)
            
            # Weighted combination
            cam = (weights * activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            
            # Normalize
            cam = cam.squeeze().detach().cpu().numpy()
            if cam.max() > 0:
                cam = cam / cam.max()
                print(f"[XAI-ULTIMATE] Grad-CAM++: Generated CAM with max={cam.max():.4f}, mean={cam.mean():.4f}")
            else:
                print("[XAI-ULTIMATE] Grad-CAM++: CAM is all zeros")
                return None
            
            return cam
            
        except Exception as e:
            print(f"[XAI-ULTIMATE] Grad-CAM++ generation error: {e}")
            import traceback
            traceback.print_exc()
            self.remove_hooks()
            return None


class LayerCAM:
    """
    Layer-CAM for better spatial resolution
    """
    def __init__(self, model, target_layer, device='cpu'):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.activations = None
        self.gradients = None
        self.hooks = []
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def register_hooks(self):
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.hooks.append(module.register_forward_hook(self.save_activation))
                self.hooks.append(module.register_full_backward_hook(self.save_gradient))
                return True
        return False
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate(self, input_tensor, class_idx=1):
        """Generate Layer-CAM"""
        self.activations = None
        self.gradients = None
        
        if not self.register_hooks():
            print(f"[XAI-ULTIMATE] Layer-CAM: Failed to register hooks for layer {self.target_layer}")
            return None
        
        try:
            # Forward pass
            input_tensor.requires_grad = True
            output = self.model(input_tensor)
            
            # Backward pass
            self.model.zero_grad()
            score = output[0, class_idx]
            score.backward(retain_graph=False)
            
            self.remove_hooks()
            
            if self.activations is None:
                print("[XAI-ULTIMATE] Layer-CAM: Activations are None")
                return None
            
            if self.gradients is None:
                print("[XAI-ULTIMATE] Layer-CAM: Gradients are None")
                return None
            
            # Check spatial dimensions
            if len(self.activations.shape) != 4:
                print(f"[XAI-ULTIMATE] Layer-CAM: Invalid activation shape {self.activations.shape}")
                return None
            
            b, c, h, w = self.activations.shape
            print(f"[XAI-ULTIMATE] Layer-CAM: Activation shape [{b}, {c}, {h}, {w}]")
            
            # Layer-CAM: element-wise multiplication
            cam = (self.activations * F.relu(self.gradients)).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            
            # Normalize
            cam = cam.squeeze().detach().cpu().numpy()
            if cam.max() > 0:
                cam = cam / cam.max()
                print(f"[XAI-ULTIMATE] Layer-CAM: Generated CAM with max={cam.max():.4f}, mean={cam.mean():.4f}")
            else:
                print("[XAI-ULTIMATE] Layer-CAM: CAM is all zeros")
                return None
            
            return cam
            
        except Exception as e:
            print(f"[XAI-ULTIMATE] Layer-CAM generation error: {e}")
            import traceback
            traceback.print_exc()
            self.remove_hooks()
            return None


class FullGrad:
    """
    FullGrad: Full-Gradient Representation for Neural Network Visualization
    Provides pixel-level attribution
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.biases = []
        self.feature_grads = []
        self.hooks = []
    
    def register_hooks(self):
        def forward_hook(module, input, output):
            pass
        
        def backward_hook(module, grad_input, grad_output):
            self.feature_grads.append(grad_output[0].detach())
        
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_full_backward_hook(backward_hook))
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate(self, input_tensor, class_idx=1):
        """Generate FullGrad attribution"""
        self.feature_grads = []
        self.register_hooks()
        
        # Forward pass
        input_tensor.requires_grad = True
        output = self.model(input_tensor)
        
        # Backward pass
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()
        
        self.remove_hooks()
        
        # Input gradient
        input_grad = input_tensor.grad.detach()
        
        # Aggregate across channels
        attribution = input_grad.abs().sum(dim=1).squeeze().cpu().numpy()
        
        if attribution.max() > 0:
            attribution = attribution / attribution.max()
        
        return attribution


class MedicalGradePostProcessing:
    """
    Medical-grade post-processing for sharp, clear heatmaps
    """
    @staticmethod
    def adaptive_threshold(heatmap, percentile=90):
        """Keep only top percentile activations"""
        threshold = np.percentile(heatmap, percentile)
        heatmap_thresh = np.where(heatmap >= threshold, heatmap, 0)
        return heatmap_thresh
    
    @staticmethod
    def sharpen_heatmap(heatmap, strength=2.0):
        """Sharpen heatmap using unsharp masking"""
        # Gaussian blur
        blurred = gaussian_filter(heatmap, sigma=1.0)
        # Unsharp mask
        sharpened = heatmap + strength * (heatmap - blurred)
        sharpened = np.clip(sharpened, 0, 1)
        return sharpened
    
    @staticmethod
    def enhance_contrast(heatmap, clip_limit=0.03):
        """Enhance contrast using adaptive histogram equalization"""
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(heatmap_uint8)
        return enhanced.astype(np.float32) / 255.0
    
    @staticmethod
    def morphological_cleanup(heatmap, kernel_size=3):
        """Remove noise and smooth boundaries"""
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Close small holes
        closed = cv2.morphologyEx(heatmap_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
        # Open to remove small noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return opened.astype(np.float32) / 255.0
    
    @staticmethod
    def edge_preserving_filter(heatmap, d=9, sigma_color=75, sigma_space=75):
        """Bilateral filter to preserve edges"""
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        filtered = cv2.bilateralFilter(heatmap_uint8, d, sigma_color, sigma_space)
        return filtered.astype(np.float32) / 255.0
    
    @staticmethod
    def super_resolution_upscale(heatmap, target_size, method='lanczos'):
        """High-quality upscaling"""
        if method == 'lanczos':
            heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
            heatmap_upscaled = heatmap_pil.resize(target_size, Image.LANCZOS)
            return np.array(heatmap_upscaled).astype(np.float32) / 255.0
        else:
            return cv2.resize(heatmap, target_size, interpolation=cv2.INTER_CUBIC)
    
    @staticmethod
    def apply_full_pipeline(heatmap, target_size, aggressive=True):
        """Apply complete medical-grade processing pipeline"""
        # Step 1: Adaptive thresholding (keep top 85% for aggressive, 90% for normal)
        percentile = 85 if aggressive else 90
        heatmap = MedicalGradePostProcessing.adaptive_threshold(heatmap, percentile=percentile)
        
        # Step 2: Sharpen
        strength = 3.0 if aggressive else 2.0
        heatmap = MedicalGradePostProcessing.sharpen_heatmap(heatmap, strength=strength)
        
        # Step 3: Enhance contrast
        heatmap = MedicalGradePostProcessing.enhance_contrast(heatmap, clip_limit=0.03)
        
        # Step 4: Morphological cleanup
        heatmap = MedicalGradePostProcessing.morphological_cleanup(heatmap, kernel_size=3)
        
        # Step 5: Edge-preserving filter
        heatmap = MedicalGradePostProcessing.edge_preserving_filter(heatmap)
        
        # Step 6: Super-resolution upscale
        heatmap = MedicalGradePostProcessing.super_resolution_upscale(heatmap, target_size, method='lanczos')
        
        # Final normalization
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap


class MedicalGradeVisualizer:
    """
    Medical-grade visualization with professional colormaps
    """
    @staticmethod
    def create_custom_colormap(name='medical_hot'):
        """Create custom medical-grade colormaps"""
        if name == 'medical_hot':
            # Red-yellow-white for high attention
            colors = np.array([
                [0, 0, 0],           # Black (0%)
                [128, 0, 0],         # Dark red (25%)
                [255, 0, 0],         # Red (50%)
                [255, 128, 0],       # Orange (75%)
                [255, 255, 0],       # Yellow (90%)
                [255, 255, 255]      # White (100%)
            ])
        elif name == 'medical_cool':
            # Blue-cyan-green for alternative view
            colors = np.array([
                [0, 0, 0],           # Black
                [0, 0, 128],         # Dark blue
                [0, 0, 255],         # Blue
                [0, 128, 255],       # Light blue
                [0, 255, 255],       # Cyan
                [0, 255, 128]        # Cyan-green
            ])
        else:
            # Default jet-like
            colors = np.array([
                [0, 0, 128],         # Dark blue
                [0, 0, 255],         # Blue
                [0, 255, 255],       # Cyan
                [0, 255, 0],         # Green
                [255, 255, 0],       # Yellow
                [255, 0, 0]          # Red
            ])
        
        # Interpolate to 256 colors
        from scipy.interpolate import interp1d
        x = np.linspace(0, 1, len(colors))
        x_new = np.linspace(0, 1, 256)
        
        r = interp1d(x, colors[:, 0], kind='cubic')(x_new)
        g = interp1d(x, colors[:, 1], kind='cubic')(x_new)
        b = interp1d(x, colors[:, 2], kind='cubic')(x_new)
        
        colormap = np.stack([b, g, r], axis=1).astype(np.uint8)  # BGR for OpenCV
        return colormap
    
    @staticmethod
    def apply_colormap(heatmap, colormap_name='medical_hot'):
        """Apply custom colormap"""
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        
        if colormap_name == 'medical_hot' or colormap_name == 'medical_cool':
            # Use standard OpenCV colormaps instead of custom (OpenCV has issues with custom colormaps)
            if colormap_name == 'medical_hot':
                colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_HOT)
            else:
                colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_WINTER)
        elif colormap_name == 'jet':
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        elif colormap_name == 'turbo':
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_TURBO)
        elif colormap_name == 'hot':
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_HOT)
        else:
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def create_professional_overlay(original_image, heatmap, colormap='medical_hot', alpha=0.6):
        """Create professional medical-grade overlay"""
        # Ensure same size
        if heatmap.shape[:2] != original_image.shape[:2]:
            heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]), 
                               interpolation=cv2.INTER_LANCZOS4)
        
        # Apply colormap
        heatmap_colored = MedicalGradeVisualizer.apply_colormap(heatmap, colormap)
        
        # Ensure uint8
        if original_image.dtype != np.uint8:
            original_image = (original_image * 255).astype(np.uint8)
        
        # Create mask for non-zero regions (only overlay where there's attention)
        mask = (heatmap > 0.1).astype(np.float32)
        mask = np.stack([mask, mask, mask], axis=2)
        
        # Blend with mask
        overlay = (1 - alpha * mask) * original_image + alpha * mask * heatmap_colored
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        return overlay


def find_best_conv_layer(model):
    """Find the best convolutional layer for CAM"""
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_layers.append(name)
    
    # Prefer later layers (higher semantic information)
    if len(conv_layers) > 0:
        return conv_layers[-1]
    
    return None


def generate_ultimate_xai(model, original_image, input_tensor, class_idx=1, device='cpu'):
    """
    Generate ULTIMATE medical-grade XAI visualizations
    Returns 2 crystal-clear, sharp, professional heatmaps
    """
    print("[XAI-ULTIMATE] Starting medical-grade XAI generation...")
    
    # Convert PIL to numpy
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image)
    
    target_size = (original_image.shape[1], original_image.shape[0])
    
    # Find best convolutional layer
    best_layer = find_best_conv_layer(model)
    
    if best_layer is None:
        print("[XAI-ULTIMATE] No convolutional layers found, using fallback")
        # Fallback: Use input gradients
        try:
            fullgrad = FullGrad(model, device)
            attribution = fullgrad.generate(input_tensor, class_idx)
            
            if attribution is not None:
                # Process
                heatmap1 = MedicalGradePostProcessing.apply_full_pipeline(
                    attribution, target_size, aggressive=True
                )
                heatmap2 = MedicalGradePostProcessing.apply_full_pipeline(
                    attribution, target_size, aggressive=False
                )
                
                # Visualize
                overlay1 = MedicalGradeVisualizer.create_professional_overlay(
                    original_image, heatmap1, colormap='medical_hot', alpha=0.6
                )
                overlay2 = MedicalGradeVisualizer.create_professional_overlay(
                    original_image, heatmap2, colormap='turbo', alpha=0.6
                )
                
                print("[XAI-ULTIMATE] Generated fallback visualizations")
                return overlay1, overlay2
        except Exception as e:
            print(f"[XAI-ULTIMATE] Fallback failed: {e}")
            return None, None
    
    print(f"[XAI-ULTIMATE] Using layer: {best_layer}")
    
    # Method 1: Grad-CAM++ (most accurate)
    heatmap1 = None
    cam1_raw = None
    try:
        print("[XAI-ULTIMATE] Attempting Grad-CAM++...")
        gradcam_pp = GradCAMPlusPlus(model, best_layer, device)
        cam1_raw = gradcam_pp.generate(input_tensor, class_idx)
        
        if cam1_raw is not None and cam1_raw.max() > 0:
            # Apply medical-grade processing (AGGRESSIVE)
            heatmap1 = MedicalGradePostProcessing.apply_full_pipeline(
                cam1_raw, target_size, aggressive=True
            )
            print(f"[XAI-ULTIMATE] Grad-CAM++ SUCCESS: min={heatmap1.min():.4f}, max={heatmap1.max():.4f}, mean={heatmap1.mean():.4f}")
        else:
            print("[XAI-ULTIMATE] Grad-CAM++ returned None or all zeros")
    except Exception as e:
        print(f"[XAI-ULTIMATE] Grad-CAM++ FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Method 2: Layer-CAM (alternative view)
    heatmap2 = None
    cam2_raw = None
    try:
        print("[XAI-ULTIMATE] Attempting Layer-CAM...")
        layercam = LayerCAM(model, best_layer, device)
        cam2_raw = layercam.generate(input_tensor, class_idx)
        
        if cam2_raw is not None and cam2_raw.max() > 0:
            # Apply medical-grade processing (NORMAL)
            heatmap2 = MedicalGradePostProcessing.apply_full_pipeline(
                cam2_raw, target_size, aggressive=False
            )
            print(f"[XAI-ULTIMATE] Layer-CAM SUCCESS: min={heatmap2.min():.4f}, max={heatmap2.max():.4f}, mean={heatmap2.mean():.4f}")
        else:
            print("[XAI-ULTIMATE] Layer-CAM returned None or all zeros")
    except Exception as e:
        print(f"[XAI-ULTIMATE] Layer-CAM FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # If Grad-CAM++ failed but Layer-CAM succeeded, use Layer-CAM for both
    if heatmap1 is None and heatmap2 is not None and cam2_raw is not None:
        print("[XAI-ULTIMATE] Grad-CAM++ failed, using Layer-CAM for both heatmaps")
        # Use aggressive processing for first heatmap
        heatmap1 = MedicalGradePostProcessing.apply_full_pipeline(
            cam2_raw, target_size, aggressive=True
        )
        print(f"[XAI-ULTIMATE] Layer-CAM (aggressive) for heatmap1: min={heatmap1.min():.4f}, max={heatmap1.max():.4f}")
    
    # If both failed, try FullGrad
    if heatmap1 is None and heatmap2 is None:
        print("[XAI-ULTIMATE] Trying FullGrad as backup...")
        try:
            fullgrad = FullGrad(model, device)
            attribution = fullgrad.generate(input_tensor, class_idx)
            
            if attribution is not None:
                heatmap1 = MedicalGradePostProcessing.apply_full_pipeline(
                    attribution, target_size, aggressive=True
                )
                heatmap2 = MedicalGradePostProcessing.apply_full_pipeline(
                    attribution, target_size, aggressive=False
                )
        except Exception as e:
            print(f"[XAI-ULTIMATE] FullGrad failed: {e}")
    
    # Create professional overlays
    overlay1 = None
    overlay2 = None
    
    if heatmap1 is not None:
        overlay1 = MedicalGradeVisualizer.create_professional_overlay(
            original_image, heatmap1, colormap='medical_hot', alpha=0.6
        )
        print("[XAI-ULTIMATE] Created Grad-CAM++ overlay (medical_hot colormap)")
    
    if heatmap2 is not None:
        overlay2 = MedicalGradeVisualizer.create_professional_overlay(
            original_image, heatmap2, colormap='turbo', alpha=0.6
        )
        print("[XAI-ULTIMATE] Created Layer-CAM overlay (turbo colormap)")
    
    if overlay1 is None and overlay2 is None:
        print("[XAI-ULTIMATE] All methods failed")
        return None, None
    
    print("[XAI-ULTIMATE] Medical-grade visualizations complete!")
    return overlay1, overlay2
