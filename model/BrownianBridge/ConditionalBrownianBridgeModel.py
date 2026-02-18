import torch
import torch.nn as nn
from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel


class ConditionalBrownianBridgeModel(BrownianBridgeModel):
    """
    Conditional Brownian Bridge Diffusion Model
    Wraps BrownianBridgeModel to condition on theta (2D one-hot encoded)
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.theta_dim = 2  # One-hot encoded [1,0] or [0,1]
        
        # Theta embedding layer
        self.theta_embedding = nn.Linear(self.theta_dim, 128)
        
        # Conditioning injection layers (FiLM)
        # Adjust 128 and layer sizes based on your network architecture
        self.film_layers = nn.ModuleList([
            nn.Linear(128, 256),  # For gamma
            nn.Linear(128, 256),  # For beta
        ])
        
    def forward(self, x, x_cond, theta=None):
        """
        Args:
            x: Original image/data
            x_cond: Condition image/data
            theta: One-hot encoded conditioning variable (batch_size, 2)
        """
        # Get theta embedding if provided
        theta_embedding = None
        if theta is not None:
            theta_embedding = torch.relu(self.theta_embedding(theta))  # (batch, 128)
        
        # Call parent forward with theta info
        loss, info = self._forward_with_conditioning(x, x_cond, theta_embedding)
        
        return loss, info
    
    def _forward_with_conditioning(self, x, x_cond, theta_embedding):
        """
        Your existing forward logic, but modified to use theta_embedding
        to modulate network activations
        """
        # Call parent's forward but with conditioning
        # This is a placeholder - you need to integrate theta_embedding 
        # into your diffusion process
        loss, info = super().forward(x, x_cond)
        
        # You can scale or modify loss based on theta if needed
        return loss, info
    
    def sample(self, x_cond, theta=None, clip_denoised=True, **kwargs):
        """
        Sample from the model conditioned on theta
        """
        theta_embedding = None
        if theta is not None:
            theta_embedding = torch.relu(self.theta_embedding(theta))
        
        # Call parent sample method but with theta awareness
        return super().sample(x_cond, clip_denoised=clip_denoised)
