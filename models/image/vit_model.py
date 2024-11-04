import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Linear(self.patch_dim, embed_dim)
    
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, "Image size must match predefined size"
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C, self.num_patches, -1)
        patches = patches.permute(0, 2, 1, 3).contiguous().view(B, self.num_patches, -1)
        return self.proj(patches)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        Q = self.query(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attention = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(attention, dim=-1)

        out = torch.matmul(attention, V).transpose(1, 2).contiguous().view(B, N, C)
        return self.fc(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, drop_rate=0.0):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        self.dropout = nn.Dropout(drop_rate)
        self.norm2 = nn.LayerNorm(embed_dim)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(drop_rate)
        )

    def forward(self, x):
        x = x + self.dropout(self.mha(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, drop_rate=0.0):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]
        return self.fc(cls_token_final)
        
if __name__ == "__main__":
    model = VisionTransformer(img_size=224, patch_size=16, in_channels=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12)
    x = torch.randn(8, 3, 224, 224)  # Batch size of 8, 3 color channels, 224x224 image size
    logits = model(x)
    print(logits.shape) 

import math

class MLPHead(nn.Module):
    def __init__(self, embed_dim, num_classes, hidden_dim=512):
        super(MLPHead, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, num_patches, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, num_patches, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.norm(x)
        return self.fc(x)

class VisionTransformerLarge(VisionTransformer):
    def __init__(self, img_size=384, patch_size=32, in_channels=3, num_classes=1000, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0, drop_rate=0.0):
        super(VisionTransformerLarge, self).__init__(img_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, mlp_ratio, drop_rate)

        self.head = ClassificationHead(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        return self.head(x[:, 0])

class VisionTransformerSmall(VisionTransformer):
    def __init__(self, img_size=128, patch_size=8, in_channels=3, num_classes=100, embed_dim=384, depth=8, num_heads=8, mlp_ratio=4.0, drop_rate=0.0):
        super(VisionTransformerSmall, self).__init__(img_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, mlp_ratio, drop_rate)

        self.head = ClassificationHead(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        return self.head(x[:, 0])

class VisionTransformerTiny(VisionTransformer):
    def __init__(self, img_size=64, patch_size=4, in_channels=3, num_classes=10, embed_dim=192, depth=4, num_heads=4, mlp_ratio=4.0, drop_rate=0.0):
        super(VisionTransformerTiny, self).__init__(img_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, mlp_ratio, drop_rate)

        self.head = ClassificationHead(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        return self.head(x[:, 0])

class PatchEmbedding3D(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, depth=8):
        super(PatchEmbedding3D, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.depth = depth
        self.proj = nn.Linear(self.patch_dim * depth, embed_dim)
    
    def forward(self, x):
        B, C, D, H, W = x.shape
        assert H == self.img_size and W == self.img_size, "Image size must match predefined size"
        assert D == self.depth, "Depth size must match predefined size"
        patches = x.unfold(3, self.patch_size, self.patch_size).unfold(4, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C, D, self.num_patches, -1)
        patches = patches.permute(0, 2, 3, 1, 4).contiguous().view(B, self.num_patches, -1)
        return self.proj(patches)

class VisionTransformer3D(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, drop_rate=0.0, time_steps=8):
        super(VisionTransformer3D, self).__init__(img_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, mlp_ratio, drop_rate)
        self.time_steps = time_steps
        self.patch_embed_3d = PatchEmbedding3D(img_size, patch_size, in_channels, embed_dim, time_steps)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed_3d(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]
        return self.fc(cls_token_final)

class VisionTransformerForSegmentation(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=21, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, drop_rate=0.0):
        super(VisionTransformerForSegmentation, self).__init__(img_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, mlp_ratio, drop_rate)
        self.segmentation_head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        x = x[:, 1:]  # Remove CLS token
        x = x.transpose(1, 2).contiguous().view(B, -1, int(H / self.patch_size), int(W / self.patch_size))
        x = F.interpolate(x, scale_factor=self.patch_size, mode="bilinear", align_corners=False)
        return self.segmentation_head(x)

class VisionTransformerForDetection(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=100, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, drop_rate=0.0):
        super(VisionTransformerForDetection, self).__init__(img_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, mlp_ratio, drop_rate)
        self.detection_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]
        return self.detection_head(cls_token_final)

class HybridVisionTransformer(VisionTransformer):
    def __init__(self, backbone, img_size=224, patch_size=16, in_channels=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, drop_rate=0.0):
        super(HybridVisionTransformer, self).__init__(img_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, mlp_ratio, drop_rate)
        self.backbone = backbone
        self.hybrid_embed = nn.Conv2d(backbone.out_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B = x.shape[0]
        x = self.backbone(x)
        x = self.hybrid_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]
        return self.fc(cls_token_final)

class VisionTransformerWithSkipConnections(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, drop_rate=0.0):
        super(VisionTransformerWithSkipConnections, self).__init__(img_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, mlp_ratio, drop_rate)
        self.skip_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for idx, block in enumerate(self.transformer_blocks):
            skip_input = x if idx % 2 == 0 else None  # Apply skip connection every other block
            x = block(x)
            if skip_input is not None:
                x = x + self.skip_proj(skip_input)

        x = self.norm(x)
        cls_token_final = x[:, 0]
        return self.fc(cls_token_final)

class DynamicVisionTransformer(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, drop_rate=0.0, dynamic_depth=True):
        super(DynamicVisionTransformer, self).__init__(img_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, mlp_ratio, drop_rate)
        self.dynamic_depth = dynamic_depth

    def forward(self, x, selected_depth=None):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        if self.dynamic_depth and selected_depth is not None:
            transformer_blocks = self.transformer_blocks[:selected_depth]
        else:
            transformer_blocks = self.transformer_blocks

        for block in transformer_blocks:
            x = block(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]
        return self.fc(cls_token_final)

class EnsembleVisionTransformers(nn.Module):
    def __init__(self, models, num_classes=1000):
        super(EnsembleVisionTransformers, self).__init__()
        self.models = nn.ModuleList(models)
        self.fc = nn.Linear(len(models) * models[0].fc.out_features, num_classes)

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        combined_output = torch.cat(outputs, dim=1)
        return self.fc(combined_output)