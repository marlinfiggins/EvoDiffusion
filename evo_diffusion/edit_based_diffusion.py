import torch
import torch.nn as nn
import torch.nn.functional as F

from evo_diffusion.torch_components import TransformerSequenceEmbedding, SinusoidalPositionalEmbedding, MLP

class EditBasedDiffusion(nn.Module):
    def __init__(self, sequence_length, num_residues, embedding_dim, transformer_params, mlp_params):
        """
        Diffusion model with shared sequence and temporal embeddings.

        Args:
            sequence_length: Length of the sequence (e.g., protein length).
            num_residues: Number of possible residues (e.g., 20 amino acids).
            embedding_dim: Dimension of the shared embeddings.
            transformer_params: Dict of params for sequence embedding transformer (keys: `num_heads`, `num_layers`, `feedforward_dim`)
            mlp_params: Dict of params for MLP forward and reverse heads (keys: `feedforward_dim`, `activation`, `dropout`)
        """
        super(EditBasedDiffusion, self).__init__()
        self.sequence_length = sequence_length
        self.num_residues = num_residues
        self.embedding_dim = embedding_dim
        self.transformer_params = transformer_params
        self.mlp_params = mlp_params

        # Shared sequence embedding network
        # self.sequence_embedding_net = nn.Sequential(
        #     nn.Linear(sequence_length * num_residues, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, embedding_dim)
        # )
        self.sequence_embedding_net = TransformerSequenceEmbedding(sequence_length, **transformer_params)

        # Shared temporal embedding network
        # self.time_embedding_net = nn.Sequential(
        #     nn.Linear(1, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, embedding_dim)
        # )
        self.time_embedding_net = SinusoidalPositionalEmbedding(embedding_dim)

        # Forward and reverse networks
        # self.forward_net = nn.Sequential(
        #     nn.Linear(embedding_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, num_residues)
        # )
        #self.forward_net = UNet(input_channels=embedding_dim, output_channels=num_residues, features=unet_features)
        self.forward_net = MLP(embedding_dim, num_residues, mlp_params)

        # self.reverse_net = nn.Sequential(
        #     nn.Linear(embedding_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, num_residues)
        # )
        #self.reverse_net = UNet(input_channels=embedding_dim, output_channels=num_residues, features=unet_features)
        self.reverse_net = MLP(embedding_dim, num_residues, mlp_params)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.forward_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4)
        self.reverse_attention = self.forward_attention
        
    def embed_sequence(self, x):
        """
        Embed the sequence and append a CLS token for global representation.
    
        Args:
            x: Input sequence (batch_size, sequence_length, num_residues).
    
        Returns:
            seq_with_cls: Sequence embedding with CLS token (batch_size, sequence_length + 1, embedding_dim).
        """
        seq_embedding = self.sequence_embedding_net(x)  # (batch_size, sequence_length, embedding_dim)
    
        # Add CLS token
        batch_size = seq_embedding.size(0)
        cls_token = self.cls_token.repeat(batch_size, 1, 1)  # (batch_size, 1, embedding_dim)
        seq_with_cls = torch.cat([cls_token, seq_embedding], dim=1)  # (batch_size, sequence_length + 1, embedding_dim)
    
        return seq_with_cls
        
    def embed_time(self, t):
        """
        Embed the time step using the shared temporal embedding network.

        Args:
            t: Time step (batch_size, 1).

        Returns:
            time_embedding: Shared time embedding (batch_size, embedding_dim).
        """
        #batch_size = t.size(0)
        #t_flat = t.view(batch_size, -1)
        time_embedding = self.time_embedding_net(t)
        return time_embedding


    def combine_embeddings(self, seq_embedding, time_embedding, attention_layer):
        """
        Combines sequence and time embeddings using attention.
    
        Args:
            seq_embedding: (batch_size, sequence_length + 1, embedding_dim), includes CLS token.
            time_embedding: (batch_size, embedding_dim).
            attention_layer: Attention module (e.g., forward_attention or reverse_attention).
    
        Returns:
            time_aware_cls: Time-aware global representation (batch_size, embedding_dim).
            updated_seq_embedding: Updated sequence embedding (batch_size, sequence_length + 1, embedding_dim).
        """
        # Transpose seq_embedding for attention (sequence_length + 1, batch_size, embedding_dim)
        seq_embedding = seq_embedding.permute(1, 0, 2)
    
        # Reshape time_embedding to match attention input (1, batch_size, embedding_dim)
        time_embedding = time_embedding.unsqueeze(0)
    
        # Apply attention: time_embedding as query, seq_embedding as key and value
        attn_output, _ = attention_layer(query=time_embedding, key=seq_embedding, value=seq_embedding)
    
        # Extract the updated CLS token (first position in sequence)
        time_aware_cls = attn_output.squeeze(0)  # (batch_size, embedding_dim)
    
        # Updated sequence embeddings (including CLS)
        updated_seq_embedding = seq_embedding.permute(1, 0, 2)  # (batch_size, sequence_length + 1, embedding_dim)
    
        return time_aware_cls, updated_seq_embedding
    
    def forward_edit_field(self, x, t, noise):
        """
        Generate the forward edit field, incorporating noise as part of the input.

        Args:
            x: Input sequence (batch_size, sequence_length, num_residues).
            t: Time step (batch_size, 1).
            noise: Noise tensor (batch_size, sequence_length, num_residues).

        Returns:
            edit_field: Probabilities of edits (batch_size, sequence_length, num_residues).
        """
        # Embed sequence and time
        seq_with_cls = self.embed_sequence(x)
        time_embedding = self.embed_time(t)

        # Inject noise into embeddings
        noise = F.pad(noise, (0, 0, 1, 0))
        noisy_seq = seq_with_cls + noise  # Add noise to sequence embedding

        # Combine sequence + noise + time embeddings using attention
        time_aware_cls, combined_embedding = self.combine_embeddings(noisy_seq, time_embedding, self.forward_attention)

        # Remove CLS token before feeding into network
        combined_embedding = combined_embedding[:, 1:, :]

        # Flatten for processing
        batch_size, sequence_length, embedding_dim = combined_embedding.shape
        reshaped_embedding = combined_embedding.reshape(batch_size * sequence_length, embedding_dim)

        # Generate edit field
        edit_field = self.forward_net(reshaped_embedding)

        return edit_field.view(-1, self.sequence_length, self.num_residues)

    def reverse_edit_field(self, x, t, noise):
        """
        Generate the reverse edit field, incorporating noise as part of the input.

        Args:
            x: Input sequence (batch_size, sequence_length, num_residues).
            t: Time step (batch_size, 1).
            noise: Noise tensor (batch_size, sequence_length, num_residues).

        Returns:
            reverse_field: Probabilities of reverse edits (batch_size, sequence_length, num_residues).
        """
        seq_with_cls = self.embed_sequence(x)
        time_embedding = self.embed_time(t)

        # Inject noise into embeddings
        noise = F.pad(noise, (0, 0, 1, 0))  
        noisy_seq = seq_with_cls + noise  # Add noise explicitly

        # Combine sequence + noise + time embeddings
        time_aware_cls, combined_embedding = self.combine_embeddings(noisy_seq, time_embedding, self.reverse_attention)

        # Remove CLS token
        combined_embedding = combined_embedding[:, 1:, :]

        # Flatten for processing
        batch_size, sequence_length, embedding_dim = combined_embedding.shape
        reshaped_embedding = combined_embedding.reshape(batch_size * sequence_length, embedding_dim)

        reverse_field = self.reverse_net(reshaped_embedding)

        return reverse_field.view(-1, self.sequence_length, self.num_residues)

    def apply_edits(self, x, edit_field, eps=1e-8):
        """
        Apply soft updates to the sequence using the edit field probabilities.
    
        Args:
            x: Input sequence (batch_size, sequence_length, num_residues).
            edit_field: Logits representing probabilities of edits (batch_size, sequence_length, num_residues).
            eps: Epsilon error to prevent undefined gradients
    
        Returns:
            x_t: Updated sequence after applying edits (soft distribution).
        """
        # Update sequence prob using predicted edit field
        x_t = F.softmax(torch.log(x + eps) + edit_field, dim=-1)
        return x_t

    def forward(self, x, t):
        """
        Forward diffusion process.

        Args:
            x: Input sequence (batch_size, sequence_length, num_residues).
            t: Time step (batch_size, 1).

        Returns:
            x_t: Sequence after applying edits.
        """
        # Sample noise
        noise = torch.randn(x.shape[0], x.shape[1], self.embedding_dim, device=x.device)  # Gaussian noise

        # Get edit field prediction with noise as input
        edit_field = self.forward_edit_field(x, t, noise)

        # Apply edits using noise-aware update
        x_t = self.apply_edits(x, edit_field)
        return x_t

    def reverse(self, x, t):
        """
        Reverse diffusion process.

        Args:
            x: Input sequence (batch_size, sequence_length, num_residues).
            t: Time step (batch_size, 1).

        Returns:
            x_t_minus_1: Sequence denoised to time t-1.
        """
        # Sample noise for the reverse process
        noise = torch.randn(x.shape[0], x.shape[1], self.embedding_dim, device=x.device)  # Gaussian noise

        # Get reverse edit field prediction with noise as input
        reverse_field = self.reverse_edit_field(x, t, noise)

        # Apply edits using noise-aware update
        x_t_minus_1 = self.apply_edits(x, reverse_field)
        return x_t_minus_1

# def sample_forward(model, x_0, t_target, steps=10):
#     """
#     Sample sequences forward in time.

#     Args:
#         model: The diffusion model.
#         x_0: Base sequence (batch_size, sequence_length, num_residues).
#         t_target: Target time to evolve the sequence to.
#         steps: Number of diffusion steps to reach t_target.

#     Returns:
#         x_t: Sequence evolved to time t_target.
#     """
#     x_t = x_0.clone()
#     for step in range(steps):
#         # Compute time for this step
#         t = torch.tensor([[step * t_target / steps]] * x_0.size(0), device=x_0.device)
#         # Forward process: Generate edit field and apply edits
#         edit_field = model.forward_edit_field(x_t, t)
#         x_t = model.apply_edits(x_t, edit_field)
#     return x_t

def guided_sample_forward_with_predictor(model, predictor, x_0, t_target, alpha=1.0, steps=10):
    """
    Guided sampling forward in time with a target predictor.

    Args:
        model: The diffusion model.
        predictor: A pretrained model or scoring function for the target property.
        x_0: Base sequence (batch_size, sequence_length, num_residues).
        t_target: Target time to evolve the sequence to.
        alpha: Strength of the bias toward the target predictor.
        steps: Number of diffusion steps to reach t_target.

    Returns:
        x_t: Sequence evolved to time t_target with guidance from the target predictor.
    """
    x_t = x_0.clone()
    for step in range(steps):
        # Compute time for this step
        t = torch.tensor([[step * t_target / steps]] * x_0.size(0), device=x_0.device)

        # Forward process: Generate edit field
        edit_field = model.forward_edit_field(x_t, t)

        # Compute target field (gradient-based guidance)
        x_t.requires_grad_(True)  # Enable gradient computation
        predictor_score = predictor(x_t).sum()  # Sum over batch for gradient computation
        predictor_score.backward()  # Backpropagate to get gradients
        target_field = x_t.grad.detach()  # Target field based on predictor gradients
        x_t.requires_grad_(False)  # Disable gradients

        # Incorporate target field into the edit field
        edit_field = edit_field + alpha * target_field

        # Sample and apply edits
        x_t = model.apply_edits(x_t, edit_field)
    return x_t