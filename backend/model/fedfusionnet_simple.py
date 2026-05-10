"""
HetFusionNet v2 - Exact architecture matching checkpoint
Swin-Small + CrossViT-15 with SE Fusion and 2-layer head
"""
import torch
import torch.nn as nn
import timm

class FedFusionNetPlus(nn.Module):
    """
    HetFusionNet v2 - Exact checkpoint architecture
    
    Components (679 tensors, 305.75 MB):
    1. Swin-ViT-Small (331 keys) - 768 dim output
    2. CrossViT-15 (340 keys) - 576 dim output (192 + 384 dual branch)
    3. SE Fusion (2 keys) - Linear(1344→84)→Linear(84→1344)
    4. Head (6 keys) - LayerNorm(1344)→Linear(1344→256)→GELU→Dropout→Linear(256→2)
    5. Aux Swin (2 keys) - LayerNorm(768)→Linear(768→2)
    6. Aux CrossViT (2 keys) - LayerNorm(576)→Linear(576→2)
    """
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        # ═══════════════════════════════════════════════════════
        # BACKBONE 1: Swin-ViT-Small (768 dim output)
        # ═══════════════════════════════════════════════════════
        self.swin = timm.create_model(
            'swin_small_patch4_window7_224',
            pretrained=False,
            num_classes=0,
            img_size=240
        )
        
        # ═══════════════════════════════════════════════════════
        # BACKBONE 2: CrossViT-15 (576 dim output - dual branch)
        # ═══════════════════════════════════════════════════════
        self.crossvit = timm.create_model(
            'crossvit_15_240',
            pretrained=False,
            num_classes=0
        )
        
        swin_dim = 768
        crossvit_dim = 576  # CrossViT-15 outputs 192 + 384 = 576
        fused_dim = 1344  # 768 + 576
        
        # ═══════════════════════════════════════════════════════
        # SE FUSION BLOCK (Change 7: GELU instead of ReLU)
        # ═══════════════════════════════════════════════════════
        self.se_fusion = nn.Module()
        self.se_fusion.se = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 16),  # 1536 → 96
            nn.GELU(),                               # GELU (not ReLU)
            nn.Linear(fused_dim // 16, fused_dim),  # 96 → 1536
            nn.Sigmoid()
        )
        
        # ═══════════════════════════════════════════════════════
        # CLASSIFICATION HEAD (Change 8: 2-layer simplified)
        # ═══════════════════════════════════════════════════════
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),                 # 0: LayerNorm(1536)
            nn.Linear(fused_dim, 256),               # 1: Linear(1536→256)
            nn.GELU(),                               # 2: GELU
            nn.Dropout(0.5),                         # 3: Dropout
            nn.Linear(256, num_classes)              # 4: Linear(256→2)
        )
        
        # ═══════════════════════════════════════════════════════
        # AUXILIARY HEADS
        # ═══════════════════════════════════════════════════════
        self.aux_swin = nn.Sequential(
            nn.LayerNorm(swin_dim),
            nn.Linear(swin_dim, num_classes)
        )
        
        self.aux_crossvit = nn.Sequential(
            nn.LayerNorm(crossvit_dim),
            nn.Linear(crossvit_dim, num_classes)
        )
    
    def forward(self, x):
        """Forward pass - inference only"""
        # Extract features from both backbones
        swin_feat = self.swin(x)        # [B, 768]
        crossvit_feat = self.crossvit(x) # [B, 768]
        
        # Concatenate features
        fused = torch.cat([swin_feat, crossvit_feat], dim=1)  # [B, 1536]
        
        # SE Fusion block
        se_weight = self.se_fusion.se(fused)
        fused = fused * se_weight
        
        # Classification head
        logits = self.head(fused)  # [B, 2]
        
        return logits
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint - handles both raw state_dict and wrapped formats"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Load weights
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        
        print(f"[OK] Checkpoint loaded: {checkpoint_path}")
        if missing:
            print(f"  [WARNING] Missing keys: {len(missing)}")
        if unexpected:
            print(f"  [WARNING] Unexpected keys: {len(unexpected)}")
        
        return missing, unexpected
