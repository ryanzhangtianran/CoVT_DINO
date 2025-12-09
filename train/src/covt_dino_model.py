import torch
import torch.nn as nn

from transformers import PaliGemmaForConditionalGeneration, Dinov2Model

class COVT_DINO(nn.Module):
    def __init__(self, PALIGEMMA_MODEL_PATH, DINO_MODEL_PATH, DTYPE, num_vis_tokens=4):
        super().__init__()
        
        # Initialize PaliGemma 3B model
        print(f"Loading PaliGemma 3B from: {PALIGEMMA_MODEL_PATH}...")
        self.paligemma = PaliGemmaForConditionalGeneration.from_pretrained(
            PALIGEMMA_MODEL_PATH,
            torch_dtype=DTYPE,
            _attn_implementation="eager"
        )
        # Initialize DINO model
        print(f"Loading DINO from: {DINO_MODEL_PATH}...")
        self.dino = Dinov2Model.from_pretrained(
            DINO_MODEL_PATH,
            torch_dtype=DTYPE
        )

        # Freeze PaliGemma and DINO parameters
        self.paligemma.requires_grad_(False)
        self.dino.requires_grad_(False)

        # Dimension config
        self.paligemma_dim = self.paligemma.config.hidden_size
        self.dino_dim = self.dino.config.hidden_size

        # Visual tokens config
        self.num_vis_tokens = num_vis_tokens
        self.num_patches = 256  # (224x224 image / 14 patch_size)

        # Projection layer
        self.input_projection = nn.Linear(self.paligemma_dim, self.paligemma_dim, dtype=DTYPE)
        # Learnable queries
        self.learnable_queries = nn.Parameter(torch.randn(1, self.num_patches, self.paligemma_dim, dtype=DTYPE))
        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.paligemma_dim, 
            num_heads=8, 
            batch_first=True,
            dtype=DTYPE
        )
        # Output projection
        self.output_projection = nn.Linear(self.paligemma_dim, self.dino_dim, dtype=DTYPE)

    def resize_token_embeddings(self, visual_tokens_size):
        self.paligemma.resize_token_embeddings(visual_tokens_size)

    def grab_dino_features(self, pixel_values):
        with torch.no_grad():
            dino_outputs = self.dino(pixel_values=pixel_values, output_hidden_states=True)
            dino_features = dino_outputs.last_hidden_state
            return dino_features[:, -1:, :]

    def forward(self, input_ids, pixel_values_paligemma, pixel_values_dino, attention_mask, target_token_ids):
        # Get DINO features
        with torch.no_grad():
            target_features = self.grab_dino_features(pixel_values_dino) # [B, 256, 768]

        # Forward
        outputs = self.paligemma(
            input_ids=input_ids,
            pixel_values=pixel_values_paligemma,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden_state = outputs.last_hidden_state[-1]

        if target_token_ids is not None:
            visual_tokens_mask = torch.isin(input_ids, target_token_ids.view(-1))
            extracted_tokens = last_hidden_state[visual_tokens_mask].view(input_ids.shape[0], self.num_vis_tokens, -1)
            visual_tokens = extracted_tokens
        
        kv = self.input_projection(visual_tokens) # [B, 4, Dim]
        batch_size = kv.shape[0]
        q = self.learnable_queries.expand(batch_size, -1, -1) # [B, 256, Dim]
        attn_output, _ = self.cross_attention(
            query=q,
            key=kv,
            value=kv
        )
        predicted_dino_features = self.output_projection(attn_output) # [B, 256, Dim]

        loss = nn.functional.mse_loss(predicted_dino_features, target_features)
        return loss
