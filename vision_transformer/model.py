import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self,img_size,patch_size,in_channels,embd_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels=in_channels,out_channels=embd_dim,
                              kernel_size=patch_size,stride=patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.norm = nn.LayerNorm(embd_dim)
    def forward(self,x):
        x = self.proj(x)  # (B,H,W)->(B, embd_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embd_dim, N)
        x = x.transpose(1, 2)  # (B, N, embd_dim)
        x = self.norm(x)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,embd_dim,num_heads=8,bias=False,attn_dropout=0.0,proj_dropout=0.0):
        super().__init__()
        assert embd_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.num_heads = num_heads
        self.head_dim = embd_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embd_dim, embd_dim * 3, bias=bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embd_dim, embd_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)
    def forward(self,x):
        B,N,C = x.shape
        qkv = self.qkv(x).reshape(B,N,3,self.num_heads,self.head_dim).permute(2,0,3,1,4)
        q,k,v =qkv[0],qkv[1],qkv[2]  # each has shape (B, num_heads, N, head_dim)
        attn = (q @ k.transpose(-2,-1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1) # (B, num_heads, N, N)
        attn = self.attn_dropout(attn) # (B, num_heads, N, N)
        out = (attn @ v).transpose(1,2).reshape(B,N,C)  # (B,num_heads,N,N) @ (B,num_heads,N,head_dim) -> 
                                                        #(B,num_heads,N,head_dim) -> (B,N,num_heads*head_dim=C)
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out
class MLP(nn.Module):
    def __init__(self,embd_dim,mlp_ratio=4.0,dropout=0.0):
        super().__init__()
        hidden_dim = int(embd_dim * mlp_ratio)
        self.fc1 = nn.Linear(embd_dim,hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim,embd_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self,embd_dim,num_heads,mlp_ratio=4.0,bias=False,
                 attn_dropout=0.0,proj_dropout=0.0,mlp_dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embd_dim)
        self.attn = MultiHeadSelfAttention(embd_dim,num_heads,bias,attn_dropout,proj_dropout)
        self.norm2 = nn.LayerNorm(embd_dim)
        self.mlp = MLP(embd_dim,mlp_ratio,mlp_dropout)
    def forward(self,x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
class VisionTransformer(nn.Module):
    def __init__(self,img_size=224,patch_size=16,in_channels=3,
                 embd_dim=768,num_heads =12,mlp_ratio=4.0,num_layers=12,
                 bias=False,attn_dropout=0.0,proj_dropout=0.0,mlp_dropout=0.0,
                 num_classes=1000):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size,patch_size,in_channels,embd_dim)
        self.cls_token = nn.Parameter(torch.zeros(1,1,embd_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1,1 + self.patch_embed.num_patches,embd_dim))
        self.pos_dropout = nn.Dropout(proj_dropout)
        self.blocks = nn.ModuleList([
            Block(embd_dim,num_heads,mlp_ratio,bias,attn_dropout,proj_dropout,mlp_dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embd_dim)
        self.head = nn.Linear(embd_dim,num_classes)
        nn.init.trunc_normal_(self.pos_embed,std=0.02)
        nn.init.trunc_normal_(self.cls_token,std=0.02)
        self.apply(self._init_weights)
    def _init_weights(self,m):
        if isinstance(m,nn.Linear):
            nn.init.trunc_normal_(m.weight,std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias,0)
            nn.init.constant_(m.weight,1.0)
    def forward(self,x):
        # (B, C, H, W)->B, N, embd_dim
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B,-1,-1)  # (1,1,embd_dim)->(B,1,embd_dim)
        x = self.patch_embed(x)  # (B, N, embd_dim)
        x = torch.cat((cls_tokens,x),dim=1)  # (B, N+1, embd_dim)
        x = x + self.pos_embed  # (B, N+1, embd_dim)
        x = self.pos_dropout(x)
        for block in self.blocks:
            x = block(x)  # (B, N+1, embd_dim)
        x = self.norm(x)  # (B, N+1, embd_dim)
        cls_output = x[:,0]  # (B, embd_dim)    
        logits = self.head(cls_output)  # (B, num_classes)
        return logits          
    
