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
        
        # # Theta embedding layer
        # self.theta_embedding = nn.Linear(self.theta_dim, 128)
        
    def forward(self, x, x_cond, theta=None):
        """
        Args:
            x: Original image/data
            x_cond: Condition image/data
            theta: One-hot encoded conditioning variable (batch_size, 2)
        """
        # Get theta embedding if provided
        # theta_embedding = None
        # if theta is not None:
        #     print("Theta is not None")
        #     theta_embedding = torch.relu(self.theta_embedding(theta))  # [batch, 128]
        #     print(f"Size of x: {x.shape}")
        # else:
        #     theta_embedding = None
        #     print("Theta is None")
        
        # loss, info = super().forward(x, x_cond, context=theta_embedding)
        loss, info = super().forward(x, x_cond, context=theta)

        return loss, info


    @torch.no_grad()
    def sample(self, x_cond, theta=None, clip_denoised=True, **kwargs):
        """
        Sample from the model conditioned on theta
        """
        # if theta is not None:
        #     theta_embedding = torch.relu(self.theta_embedding(theta))            
        # else:
        #     theta_embedding = None

        return super().sample(x_cond, context=theta, clip_denoised=clip_denoised, **kwargs)
