import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import pytorch_lightning as pl
from einops import rearrange, repeat
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        return torch.matmul(attention_weights, V)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        return self.w_o(attention_output)


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward network"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class ModernPoseForecaster(pl.LightningModule):
    """
    Modern transformer-based pose forecasting model.
    
    This model uses:
    - Transformer architecture for sequence modeling
    - Multi-head attention for capturing temporal dependencies
    - Positional encoding for sequence order
    - Residual connections and layer normalization
    
    Args:
        input_dim (int): Number of pose keypoints * 2 (x, y coordinates)
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        num_layers (int): Number of transformer layers
        d_ff (int): Feed-forward dimension
        dropout (float): Dropout rate
        max_seq_len (int): Maximum sequence length
        lr (float): Learning rate
    """
    
    def __init__(
        self,
        input_dim: int = 34,  # 17 keypoints * 2 coordinates
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 100,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, input_dim)
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
    def create_causal_mask(self, seq_len: int) -> Tensor:
        """Create causal mask for autoregressive generation"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask == 0
        
    def forward(self, x: Tensor, max_length: int = 10) -> Tensor:
        """
        Forward pass for pose forecasting
        
        Args:
            x: Input sequence of shape (batch_size, seq_len, input_dim)
            max_length: Maximum number of frames to predict
            
        Returns:
            Predicted pose sequence of shape (batch_size, max_length, input_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Create causal mask for autoregressive generation
        mask = self.create_causal_mask(seq_len).to(x.device)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, mask)
        
        # Generate future frames autoregressively
        predictions = []
        current_input = x
        
        for _ in range(max_length):
            # Get the last frame's representation
            last_frame = current_input[:, -1:, :]
            
            # Project to pose coordinates
            predicted_pose = self.output_projection(last_frame)
            predictions.append(predicted_pose)
            
            # Add the predicted frame to the sequence for next iteration
            predicted_encoded = self.input_projection(predicted_pose)
            current_input = torch.cat([current_input, predicted_encoded], dim=1)
            
            # Update mask for the new sequence length
            new_seq_len = current_input.size(1)
            mask = self.create_causal_mask(new_seq_len).to(x.device)
            
            # Re-apply transformer layers
            temp_input = current_input
            for layer in self.transformer_layers:
                temp_input = layer(temp_input, mask)
            current_input = temp_input
        
        return torch.cat(predictions, dim=1)
    
    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """Training step with teacher forcing"""
        # batch shape: (batch_size, seq_len, input_dim)
        input_seq = batch[:, :-10, :]  # All but last 10 frames
        target_seq = batch[:, -10:, :]  # Last 10 frames
        
        predictions = self.forward(input_seq, max_length=10)
        
        # Calculate losses
        mse_loss = self.mse_loss(predictions, target_seq)
        mae_loss = self.mae_loss(predictions, target_seq)
        
        # Combined loss
        total_loss = mse_loss + 0.1 * mae_loss
        
        # Log metrics
        self.log('train_mse_loss', mse_loss, prog_bar=True)
        self.log('train_mae_loss', mae_loss, prog_bar=True)
        self.log('train_total_loss', total_loss, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """Validation step"""
        input_seq = batch[:, :-10, :]
        target_seq = batch[:, -10:, :]
        
        predictions = self.forward(input_seq, max_length=10)
        
        mse_loss = self.mse_loss(predictions, target_seq)
        mae_loss = self.mae_loss(predictions, target_seq)
        
        # Calculate MPJPE (Mean Per Joint Position Error)
        mpjpe = self.calculate_mpjpe(predictions, target_seq)
        
        total_loss = mse_loss + 0.1 * mae_loss
        
        self.log('val_mse_loss', mse_loss, prog_bar=True)
        self.log('val_mae_loss', mae_loss, prog_bar=True)
        self.log('val_mpjpe', mpjpe, prog_bar=True)
        self.log('val_total_loss', total_loss, prog_bar=True)
        
        return total_loss
    
    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """Test step with comprehensive evaluation"""
        input_seq = batch[:, :-10, :]
        target_seq = batch[:, -10:, :]
        
        predictions = self.forward(input_seq, max_length=10)
        
        # Calculate all metrics
        mse_loss = self.mse_loss(predictions, target_seq)
        mae_loss = self.mae_loss(predictions, target_seq)
        mpjpe = self.calculate_mpjpe(predictions, target_seq)
        pdj_score = self.calculate_pdj(predictions, target_seq)
        
        self.log('test_mse_loss', mse_loss)
        self.log('test_mae_loss', mae_loss)
        self.log('test_mpjpe', mpjpe)
        self.log('test_pdj_score', pdj_score)
        
        return mse_loss
    
    def calculate_mpjpe(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Calculate Mean Per Joint Position Error"""
        # Reshape to (batch_size * frames, num_joints, 2)
        pred_reshaped = predictions.view(-1, 17, 2)
        target_reshaped = targets.view(-1, 17, 2)
        
        # Calculate Euclidean distance for each joint
        distances = torch.norm(pred_reshaped - target_reshaped, dim=-1)
        
        # Return mean across all joints and frames
        return torch.mean(distances)
    
    def calculate_pdj(self, predictions: Tensor, targets: Tensor, threshold: float = 0.2) -> Tensor:
        """Calculate Percentage of Detected Joints"""
        # Reshape to (batch_size * frames, num_joints, 2)
        pred_reshaped = predictions.view(-1, 17, 2)
        target_reshaped = targets.view(-1, 17, 2)
        
        # Calculate distances
        distances = torch.norm(pred_reshaped - target_reshaped, dim=-1)
        
        # Calculate torso diameter for normalization
        pelvis = target_reshaped[:, 0]  # First joint (pelvis)
        shoulder_center = target_reshaped[:, 8]  # 9th joint (shoulder center)
        torso_diameter = torch.norm(pelvis - shoulder_center, dim=-1)
        
        # Threshold is 20% of torso diameter
        thresholds = torso_diameter * threshold
        
        # Count joints within threshold
        detected_joints = (distances < thresholds.unsqueeze(1)).float()
        
        # Return percentage of detected joints
        return torch.mean(detected_joints)
    
    def predict_sequence(self, input_sequence: Tensor, num_frames: int = 10) -> Tensor:
        """Predict future pose sequence"""
        self.eval()
        with torch.no_grad():
            predictions = self.forward(input_sequence, max_length=num_frames)
        return predictions
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_total_loss",
            },
        } 