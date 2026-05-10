"""
Whole-Slide Imaging (WSI) Processor
Handles gigapixel pathology images with tile-based processing and attention aggregation
"""

import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path
import cv2

# ============================================
# WSI CONFIGURATION
# ============================================

WSI_TILE_SIZE = 256  # Tile size for extraction
WSI_OVERLAP = 0  # Overlap between tiles (0 = no overlap)
WSI_MAGNIFICATION = 20  # Target magnification (20x)
WSI_TISSUE_THRESHOLD = 0.5  # Minimum tissue content per tile

# ============================================
# WSI PROCESSOR CLASS
# ============================================

class WSIProcessor:
    """
    Whole-Slide Image Processor
    Handles .svs, .ndpi, .tiff, and other WSI formats
    """
    
    def __init__(self, tile_size: int = WSI_TILE_SIZE, overlap: int = WSI_OVERLAP):
        """
        Initialize WSI processor
        
        Args:
            tile_size: Size of extracted tiles (default: 256)
            overlap: Overlap between tiles in pixels (default: 0)
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.openslide_available = False
        self.large_image_available = False
        
        # Try to import OpenSlide
        try:
            import openslide
            self.openslide = openslide
            self.openslide_available = True
            print("[OK] OpenSlide library loaded")
        except ImportError:
            print("[WARNING] OpenSlide not installed. WSI support limited.")
            print("  Install: pip install openslide-python")
        
        # Try to import large_image
        try:
            import large_image
            self.large_image = large_image
            self.large_image_available = True
            print("[OK] large_image library loaded")
        except ImportError:
            print("[WARNING] large_image not installed. Some WSI formats may not be supported.")
            print("  Install: pip install large-image[openslide]")
    
    def is_wsi_file(self, file_path: str) -> bool:
        """
        Check if file is a WSI format
        
        Args:
            file_path: Path to image file
            
        Returns:
            True if WSI format, False otherwise
        """
        wsi_extensions = ['.svs', '.ndpi', '.tiff', '.tif', '.scn', '.mrxs', '.vms', '.vmu', '.bif']
        ext = Path(file_path).suffix.lower()
        return ext in wsi_extensions
    
    def load_wsi(self, file_path: str) -> Optional[Any]:
        """
        Load WSI file
        
        Args:
            file_path: Path to WSI file
            
        Returns:
            OpenSlide object or None if failed
        """
        if not self.openslide_available:
            print("[ERROR] OpenSlide not available. Cannot load WSI.")
            return None
        
        try:
            slide = self.openslide.OpenSlide(file_path)
            print(f"[WSI] Loaded: {file_path}")
            print(f"[WSI] Dimensions: {slide.dimensions}")
            print(f"[WSI] Levels: {slide.level_count}")
            print(f"[WSI] Magnification: {slide.properties.get('openslide.objective-power', 'Unknown')}")
            return slide
        except Exception as e:
            print(f"[ERROR] Failed to load WSI: {e}")
            return None
    
    def get_tissue_mask(self, slide: Any, level: int = -1) -> np.ndarray:
        """
        Generate tissue mask to identify tissue regions
        
        Args:
            slide: OpenSlide object
            level: Pyramid level (-1 = lowest resolution)
            
        Returns:
            Binary mask (1 = tissue, 0 = background)
        """
        try:
            # Get thumbnail at lowest resolution
            if level == -1:
                level = slide.level_count - 1
            
            # Read thumbnail
            thumbnail = slide.read_region((0, 0), level, slide.level_dimensions[level])
            thumbnail = np.array(thumbnail.convert('RGB'))
            
            # Convert to grayscale
            gray = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2GRAY)
            
            # Otsu thresholding
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Morphological operations to clean mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Convert to binary
            mask = (mask > 0).astype(np.uint8)
            
            print(f"[WSI] Tissue mask generated: {mask.shape}")
            print(f"[WSI] Tissue coverage: {mask.sum() / mask.size * 100:.2f}%")
            
            return mask
            
        except Exception as e:
            print(f"[ERROR] Failed to generate tissue mask: {e}")
            return None
    
    def extract_tiles(
        self,
        slide: Any,
        level: int = 0,
        tissue_threshold: float = WSI_TISSUE_THRESHOLD
    ) -> List[Dict[str, Any]]:
        """
        Extract tiles from WSI
        
        Args:
            slide: OpenSlide object
            level: Pyramid level to extract from (0 = highest resolution)
            tissue_threshold: Minimum tissue content (0.0-1.0)
            
        Returns:
            List of tile dictionaries with image data and coordinates
        """
        try:
            # Get dimensions
            width, height = slide.level_dimensions[level]
            
            # Generate tissue mask
            tissue_mask = self.get_tissue_mask(slide)
            if tissue_mask is None:
                print("[WARNING] No tissue mask available. Extracting all tiles.")
                tissue_mask = np.ones((height // self.tile_size, width // self.tile_size), dtype=np.uint8)
            
            # Resize mask to match tile grid
            mask_height = height // self.tile_size
            mask_width = width // self.tile_size
            tissue_mask_resized = cv2.resize(tissue_mask, (mask_width, mask_height), interpolation=cv2.INTER_NEAREST)
            
            # Extract tiles
            tiles = []
            stride = self.tile_size - self.overlap
            
            for y in range(0, height - self.tile_size + 1, stride):
                for x in range(0, width - self.tile_size + 1, stride):
                    # Check tissue content
                    mask_y = y // self.tile_size
                    mask_x = x // self.tile_size
                    
                    if mask_y < tissue_mask_resized.shape[0] and mask_x < tissue_mask_resized.shape[1]:
                        tissue_content = tissue_mask_resized[mask_y, mask_x]
                        
                        if tissue_content < tissue_threshold:
                            continue  # Skip tiles with insufficient tissue
                    
                    # Extract tile
                    tile = slide.read_region((x, y), level, (self.tile_size, self.tile_size))
                    tile = tile.convert('RGB')
                    tile_array = np.array(tile)
                    
                    # Store tile info
                    tiles.append({
                        'image': tile_array,
                        'coordinates': (x, y),
                        'level': level,
                        'tissue_content': float(tissue_content) if tissue_content is not None else 1.0
                    })
            
            print(f"[WSI] Extracted {len(tiles)} tiles from {width}x{height} image")
            return tiles
            
        except Exception as e:
            print(f"[ERROR] Failed to extract tiles: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def preprocess_tile(self, tile: np.ndarray) -> torch.Tensor:
        """
        Preprocess tile for model input
        
        Args:
            tile: Tile image (numpy array)
            
        Returns:
            Preprocessed tensor
        """
        from torchvision import transforms
        
        # Convert to PIL Image
        tile_pil = Image.fromarray(tile)
        
        # Apply same preprocessing as regular images
        transform = transforms.Compose([
            transforms.Resize((240, 240)),  # Match model input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transform(tile_pil)
    
    def aggregate_tile_predictions(
        self,
        tile_predictions: List[Dict[str, Any]],
        method: str = 'attention'
    ) -> Dict[str, Any]:
        """
        Aggregate predictions from multiple tiles
        
        Args:
            tile_predictions: List of tile prediction dictionaries
            method: Aggregation method ('max', 'mean', 'attention')
            
        Returns:
            Aggregated prediction
        """
        if not tile_predictions:
            return None
        
        # Extract probabilities
        probs = np.array([pred['probability'] for pred in tile_predictions])
        classes = np.array([pred['class'] for pred in tile_predictions])
        
        if method == 'max':
            # Max pooling
            max_idx = np.argmax(probs)
            return {
                'class': int(classes[max_idx]),
                'probability': float(probs[max_idx]),
                'method': 'max_pooling',
                'num_tiles': len(tile_predictions)
            }
        
        elif method == 'mean':
            # Mean pooling
            mean_prob = np.mean(probs)
            mean_class = 1 if mean_prob > 0.5 else 0
            return {
                'class': mean_class,
                'probability': float(mean_prob),
                'method': 'mean_pooling',
                'num_tiles': len(tile_predictions)
            }
        
        elif method == 'attention':
            # Attention-weighted aggregation
            # Use probabilities as attention weights
            attention_weights = probs / probs.sum()
            weighted_prob = np.sum(probs * attention_weights)
            weighted_class = 1 if weighted_prob > 0.5 else 0
            
            return {
                'class': weighted_class,
                'probability': float(weighted_prob),
                'method': 'attention_weighted',
                'num_tiles': len(tile_predictions),
                'attention_weights': attention_weights.tolist()
            }
        
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def process_wsi_with_model(
        self,
        wsi_path: str,
        model: nn.Module,
        device: str = 'cpu',
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Process entire WSI with CNN model
        
        Args:
            wsi_path: Path to WSI file
            model: PyTorch model
            device: 'cpu' or 'cuda'
            batch_size: Batch size for inference
            
        Returns:
            WSI-level prediction with tile-level details
        """
        try:
            # Load WSI
            slide = self.load_wsi(wsi_path)
            if slide is None:
                return None
            
            # Extract tiles
            tiles = self.extract_tiles(slide, level=0)
            if not tiles:
                print("[ERROR] No tiles extracted from WSI")
                return None
            
            print(f"[WSI] Processing {len(tiles)} tiles with model...")
            
            # Process tiles in batches
            tile_predictions = []
            model.eval()
            
            for i in range(0, len(tiles), batch_size):
                batch_tiles = tiles[i:i+batch_size]
                
                # Preprocess batch
                batch_tensors = torch.stack([
                    self.preprocess_tile(tile['image']) for tile in batch_tiles
                ])
                batch_tensors = batch_tensors.to(device)
                
                # Inference
                with torch.no_grad():
                    outputs = model(batch_tensors)
                    probs = torch.softmax(outputs, dim=1)
                    probs_np = probs.cpu().numpy()
                
                # Store predictions
                for j, tile in enumerate(batch_tiles):
                    tile_predictions.append({
                        'coordinates': tile['coordinates'],
                        'class': int(probs_np[j].argmax()),
                        'probability': float(probs_np[j].max()),
                        'tissue_content': tile['tissue_content']
                    })
                
                if (i + batch_size) % 100 == 0:
                    print(f"[WSI] Processed {min(i + batch_size, len(tiles))}/{len(tiles)} tiles")
            
            # Aggregate predictions
            aggregated = self.aggregate_tile_predictions(tile_predictions, method='attention')
            
            # Add tile-level details
            aggregated['tile_predictions'] = tile_predictions
            aggregated['wsi_dimensions'] = slide.dimensions
            aggregated['wsi_path'] = wsi_path
            
            print(f"[WSI] Final prediction: Class {aggregated['class']} ({aggregated['probability']:.2%})")
            
            # Close slide
            slide.close()
            
            return aggregated
            
        except Exception as e:
            print(f"[ERROR] WSI processing failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_wsi_heatmap(
        self,
        wsi_result: Dict[str, Any],
        output_path: str = None
    ) -> Optional[np.ndarray]:
        """
        Generate heatmap visualization of tile predictions
        
        Args:
            wsi_result: Result from process_wsi_with_model
            output_path: Optional path to save heatmap
            
        Returns:
            Heatmap image (numpy array)
        """
        try:
            tile_predictions = wsi_result['tile_predictions']
            
            # Get grid dimensions
            coords = np.array([pred['coordinates'] for pred in tile_predictions])
            max_x = coords[:, 0].max() + self.tile_size
            max_y = coords[:, 1].max() + self.tile_size
            
            # Create heatmap
            heatmap = np.zeros((max_y // self.tile_size, max_x // self.tile_size), dtype=np.float32)
            
            for pred in tile_predictions:
                x, y = pred['coordinates']
                grid_x = x // self.tile_size
                grid_y = y // self.tile_size
                heatmap[grid_y, grid_x] = pred['probability']
            
            # Resize for visualization
            heatmap_resized = cv2.resize(heatmap, (800, 800), interpolation=cv2.INTER_LINEAR)
            
            # Apply colormap
            heatmap_colored = cv2.applyColorMap(
                (heatmap_resized * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            
            # Save if path provided
            if output_path:
                cv2.imwrite(output_path, heatmap_colored)
                print(f"[WSI] Heatmap saved: {output_path}")
            
            return heatmap_colored
            
        except Exception as e:
            print(f"[ERROR] Failed to generate heatmap: {e}")
            return None


# ============================================
# ATTENTION-BASED AGGREGATION MODEL (CLAM)
# ============================================

class AttentionAggregator(nn.Module):
    """
    Attention-based Multiple Instance Learning (MIL) aggregator
    Based on CLAM (Clustering-constrained Attention MIL)
    """
    
    def __init__(self, feature_dim: int = 1536, num_classes: int = 2):
        """
        Initialize attention aggregator
        
        Args:
            feature_dim: Dimension of tile features
            num_classes: Number of output classes
        """
        super().__init__()
        
        # Attention network
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, tile_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            tile_features: Tensor of shape (num_tiles, feature_dim)
            
        Returns:
            (predictions, attention_weights)
        """
        # Compute attention weights
        attention_scores = self.attention(tile_features)  # (num_tiles, 1)
        attention_weights = torch.softmax(attention_scores, dim=0)  # (num_tiles, 1)
        
        # Weighted aggregation
        aggregated_features = torch.sum(tile_features * attention_weights, dim=0, keepdim=True)  # (1, feature_dim)
        
        # Classification
        predictions = self.classifier(aggregated_features)  # (1, num_classes)
        
        return predictions, attention_weights


# ============================================
# FACTORY FUNCTION
# ============================================

def create_wsi_processor(tile_size: int = 256, overlap: int = 0) -> WSIProcessor:
    """
    Factory function to create WSI processor
    
    Args:
        tile_size: Size of extracted tiles
        overlap: Overlap between tiles
        
    Returns:
        WSIProcessor instance
    """
    return WSIProcessor(tile_size=tile_size, overlap=overlap)


# ============================================
# TESTING
# ============================================

if __name__ == "__main__":
    print("Testing WSI Processor...")
    
    processor = create_wsi_processor()
    
    if processor.openslide_available:
        print("✓ OpenSlide is available")
        print("  WSI formats supported: .svs, .ndpi, .tiff, .scn, .mrxs")
    else:
        print("✗ OpenSlide not available")
        print("  Install: pip install openslide-python")
    
    if processor.large_image_available:
        print("✓ large_image is available")
    else:
        print("✗ large_image not available")
        print("  Install: pip install large-image[openslide]")
