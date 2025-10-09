import torch
import torch.nn as nn
import torch.nn.functional as F

from evo_diffusion.torch_components import (MLP, SinusoidalPositionalEmbedding,
                                            TransformerSequenceEmbedding)

from evo_diffusion.lyra import PGCEmbedding


class EditBasedDiffusion(nn.Module):
    def __init__(
        self,
        sequence_length,
        num_residues,
        embedding_dim,
        transformer_params,
        mlp_params,
    ):
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
        self.sequence_embedding_net = TransformerSequenceEmbedding(
            sequence_length, **transformer_params
        )

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
        # self.forward_net = UNet(input_channels=embedding_dim, output_channels=num_residues, features=unet_features)
        self.forward_net = MLP(embedding_dim, num_residues, mlp_params)

        # self.reverse_net = nn.Sequential(
        #     nn.Linear(embedding_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, num_residues)
        # )
        # self.reverse_net = UNet(input_channels=embedding_dim, output_channels=num_residues, features=unet_features)
        self.reverse_net = MLP(embedding_dim, num_residues, mlp_params)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.forward_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=4
        )
        self.reverse_attention = self.forward_attention

    def embed_sequence(self, x):
        """
        Embed the sequence and append a CLS token for global representation.

        Args:
            x: Input sequence (batch_size, sequence_length, num_residues).

        Returns:
            seq_with_cls: Sequence embedding with CLS token (batch_size, sequence_length + 1, embedding_dim).
        """
        seq_embedding = self.sequence_embedding_net(
            x
        )  # (batch_size, sequence_length, embedding_dim)

        # Add CLS token
        batch_size = seq_embedding.size(0)
        cls_token = self.cls_token.repeat(
            batch_size, 1, 1
        )  # (batch_size, 1, embedding_dim)
        seq_with_cls = torch.cat(
            [cls_token, seq_embedding], dim=1
        )  # (batch_size, sequence_length + 1, embedding_dim)

        return seq_with_cls

    def embed_time(self, t):
        """
        Embed the time step using the shared temporal embedding network.

        Args:
            t: Time step (batch_size, 1).

        Returns:
            time_embedding: Shared time embedding (batch_size, embedding_dim).
        """
        # batch_size = t.size(0)
        # t_flat = t.view(batch_size, -1)
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
        attn_output, _ = attention_layer(
            query=time_embedding, key=seq_embedding, value=seq_embedding
        )

        # Extract the updated CLS token (first position in sequence)
        time_aware_cls = attn_output.squeeze(0)  # (batch_size, embedding_dim)

        # Updated sequence embeddings (including CLS)
        updated_seq_embedding = seq_embedding.permute(
            1, 0, 2
        )  # (batch_size, sequence_length + 1, embedding_dim)

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
        time_aware_cls, combined_embedding = self.combine_embeddings(
            noisy_seq, time_embedding, self.forward_attention
        )

        # Remove CLS token before feeding into network
        combined_embedding = combined_embedding[:, 1:, :]

        # Flatten for processing
        batch_size, sequence_length, embedding_dim = combined_embedding.shape
        reshaped_embedding = combined_embedding

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
        # noise = F.pad(noise, (0, 0, 1, 0))
        # noisy_seq = seq_with_cls + noise  # Add noise explicitly

        # Combine sequence + noise + time embeddings
        time_aware_cls, combined_embedding = self.combine_embeddings(
            seq_with_cls, time_embedding, self.reverse_attention
        )

        # Remove CLS token
        combined_embedding = combined_embedding[:, 1:, :]

        # Flatten for processing
        batch_size, sequence_length, embedding_dim = combined_embedding.shape
        reshaped_embedding = combined_embedding

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
        x_logits = torch.log(x + eps)
        x_t = F.softmax(x_logits + edit_field, dim=-1)
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
        noise = torch.randn(
            x.shape[0], x.shape[1], self.embedding_dim, device=x.device
        )  # Gaussian noise

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
        noise = torch.randn(
            x.shape[0], x.shape[1], self.embedding_dim, device=x.device
        )  # Gaussian noise

        # Get reverse edit field prediction with noise as input
        reverse_field = self.reverse_edit_field(x, t, noise)

        # Apply edits using noise-aware update
        x_t_minus_1 = self.apply_edits(x, reverse_field)
        return x_t_minus_1


def guided_sample_forward_with_predictor(
    model, predictor, x_0, t_target, alpha=1.0, steps=10
):
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
        predictor_score = predictor(
            x_t
        ).sum()  # Sum over batch for gradient computation
        predictor_score.backward()  # Backpropagate to get gradients
        target_field = x_t.grad.detach()  # Target field based on predictor gradients
        x_t.requires_grad_(False)  # Disable gradients

        # Incorporate target field into the edit field
        edit_field = edit_field + alpha * target_field

        # Sample and apply edits
        x_t = model.apply_edits(x_t, edit_field)
    return x_t

class SelectionModel(nn.Module):
    def __init__(self, num_residues, embedding_dim, time_embedding_dim):
        """
        Sequence- and time-dependent selection using dot-product similarity.

        Args:
            num_residues (int): Number of possible residues.
            embedding_dim (int): Dimensionality of context embeddings.
            time_embedding_dim (int): Dimensionality of time embedding.
        """
        super().__init__()
        self.residue_embeddings = nn.Parameter(torch.randn(num_residues, embedding_dim))
        self.time_embed = SinusoidalPositionalEmbedding(time_embedding_dim)
        self.time_proj = nn.Linear(time_embedding_dim, embedding_dim)
        self.context_proj = nn.Linear(embedding_dim + embedding_dim, embedding_dim)

    def forward(self, site_context, t):
        """
        Compute selection logits for each residue at each site.

        Args:
            site_context (Tensor): (B, L, D) — sequence-dependent context (e.g., attention queries).
            t (Tensor): (B, 1) — time input.

        Returns:
            Tensor: (B, L, N) — selection logits over target residues.
        """
        time_embedding = self.time_proj(self.time_embed(t))          # (B, D)
        time_embedding = time_embedding.unsqueeze(1).expand_as(site_context)  # (B, L, D)

        combined = torch.cat([site_context, time_embedding], dim=-1) # (B, L, 2D)
        context = self.context_proj(combined)                        # (B, L, D)

        logits = torch.einsum("bld,nd->bln", context, self.residue_embeddings)  # (B, L, N)
        return logits

class MixtureOfMutations(nn.Module):
    def __init__(
        self,
        sequence_length,
        num_residues,
        embedding_dim,
        pos_embedding_dim,
        num_basis,
        transformer_params,
        mlp_params,
        routing_mode="context",  # "sitewise", "context", "attention"
        basis_mode="time_dependent",  # or "fixed"
        selection_model=None,
    ):
        """
        Initializes a mutation model where each site is updated via a
        soft mixture of shared basis mutation matrices.

        The model supports multiple routing strategies and basis configurations:

        Routing Modes:
            - "sitewise": routing weights are based only on positional identity.
            - "context": routing weights are computed using both position and local sequence context.

        Basis Modes:
            - "fixed": basis matrices are learned and constant across time.
            - "time_dependent": basis matrices are generated from a time embedding and change with t.

        Args:
            sequence_length (int): Length of the sequence (e.g., number of amino acid sites).
            num_residues (int): Number of residue types (e.g., 20 amino acids).
            embedding_dim (int): Dimensionality of time and sequence embeddings.
            pos_embedding_dim (int): Dimensionality of position embeddings.
            num_basis (int): Number of shared mutation basis matrices.
            transformer_params (dict): Parameters for the Transformer sequence embedding.
            mlp_params (dict): Parameters for all MLPs (e.g., feedforward_dim, activation, dropout).
            routing_mode (str): How basis routing weights are computed. Options: {"sitewise", "context"}.
            basis_mode (str): How basis matrices are parameterized. Options: {"fixed", "time_dependent"}.
        """
        super(MixtureOfMutations, self).__init__()
        self.sequence_length = sequence_length
        self.num_residues = num_residues
        self.embedding_dim = embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
        self.position_embeddings = nn.Embedding(
            self.sequence_length, self.pos_embedding_dim
        )
        self.num_basis = num_basis
        self.transformer_params = transformer_params
        self.mlp_params = mlp_params
        self.routing_mode = routing_mode
        self.basis_mode = basis_mode

        # Shared sequence embedding network
        self.sequence_embedding_net = TransformerSequenceEmbedding(
            self.sequence_length, **transformer_params
        )
        # self.sequence_embedding_net = PGCEmbedding(self.sequence_length, self.num_residues, self.embedding_dim, num_layers=2)

        # Shared temporal embedding network
        self.time_embedding_net = SinusoidalPositionalEmbedding(self.embedding_dim)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        # Setting up routing weights
        if routing_mode == "attention":
            self.basis_keys = nn.Parameter(torch.randn(num_basis, embedding_dim))
            self.routing_temp = nn.Parameter(torch.tensor(1.0))

            # Projection for attention queries (sequence + position)
            self.attn_query_proj = nn.Linear(
                embedding_dim + pos_embedding_dim, embedding_dim
            )
        else:
            if self.routing_mode == "sitewise":
                weight_input_dim = self.pos_embedding_dim
            elif self.routing_mode == "context":
                weight_input_dim = self.embedding_dim + self.pos_embedding_dim
            else:
                raise ValueError(f"Unsupported routing_mode: {self.routing_mode}")
            self.site_weights_net = MLP(weight_input_dim, self.num_basis, mlp_params)

        # Setting up basis matrices
        if self.basis_mode == "fixed":
            self.basis_logits = nn.Parameter(
                torch.randn(self.num_basis, self.num_residues, self.num_residues)
            )
        elif self.basis_mode == "time_dependent":
            self.basis_generator = MLP(
                self.embedding_dim,
                self.num_basis * self.num_residues * self.num_residues,
                mlp_params,
            )
        else:
            raise ValueError(f"Unsupported basis_mode: {self.basis_mode}")


        self.selection_model = selection_model  # Pass in externally
        if self.selection_model is not None:
            # Projection for attention queries (sequence + position)
            # self.attn_query_proj = nn.Linear(
            #     embedding_dim + pos_embedding_dim, embedding_dim
            # )
            self.apply_selection = True  # Add toggle if desired

    def embed_sequence(self, x):
        """
        Embed the sequence and append a CLS token for global representation.

        Args:
            x: Input sequence (batch_size, sequence_length, num_residues).

        Returns:
            seq_with_cls: Sequence embedding with CLS token (batch_size, sequence_length + 1, embedding_dim).
        """
        seq_embedding = self.sequence_embedding_net(
            x
        )  # (batch_size, sequence_length, embedding_dim)

        # Add CLS token
        batch_size = seq_embedding.size(0)
        cls_token = self.cls_token.repeat(
            batch_size, 1, 1
        )  # (batch_size, 1, embedding_dim)
        seq_with_cls = torch.cat(
            [cls_token, seq_embedding], dim=1
        )  # (batch_size, sequence_length + 1, embedding_dim)

        return seq_with_cls

    def embed_time(self, t):
        """
        Embed the time step using the shared temporal embedding network.

        Args:
            t: Time step (batch_size, 1).

        Returns:
            time_embedding: Shared time embedding (batch_size, embedding_dim).
        """
        time_embedding = self.time_embedding_net(t)
        return time_embedding

    def forward_edit_field(self, x, t):
        # Embed sequence and time
        # seq_with_cls = self.embed_sequence(x)
        time_embedding = self.embed_time(t)
        # time_embedding = time_embedding.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, L, D_pos)

        # Get positional embeddings for each position
        positions = torch.arange(self.sequence_length, device=x.device)  # (L,)
        pos_emb = self.position_embeddings(positions)  # (L, D_pos)
        pos_emb = pos_emb.unsqueeze(0).expand(x.size(0), -1, -1)  # (B, L, D_pos)

        # Compute weights for sites
        if self.routing_mode == "sitewise":
            # Only use position embeddings
            site_weights = self.site_weights_net(pos_emb)
        elif self.routing_mode == "context":
            # Full context-aware embedding
            seq_with_cls = self.embed_sequence(x)
            mlp_input = torch.cat([seq_with_cls[:, 1:, :], pos_emb], dim=-1)
            site_weights = self.site_weights_net(mlp_input)
        elif self.routing_mode == "attention":
            seq_embed = self.embed_sequence(x)[:, 1:, :]  # (B, L, D_seq)

            # Concatenate and project to query space
            attn_input = torch.cat(
                [seq_embed, pos_emb], dim=-1
            )  # (B, L, D_seq + D_pos)
            queries = self.attn_query_proj(attn_input)  # (B, L, D)

            # Compute attention over basis keys
            logits = torch.einsum("bld,kd->blk", queries, self.basis_keys)  # (B, L, K)
            site_weights = logits / self.routing_temp
        else:
            raise ValueError(f"Unsupported routing_mode: {self.routing_mode}")
        site_weights = torch.softmax(site_weights, dim=-1)  # (L, K)

        # Compute basis mutation matrices
        if self.basis_mode == "time_dependent":
            basis_logits = self.basis_generator(time_embedding).view(
                -1, self.num_basis, self.num_residues, self.num_residues
            )
        elif self.basis_mode == "fixed":
            basis_logits = self.basis_logits.unsqueeze(0).expand(x.size(0), -1, -1, -1)
        else:
            raise ValueError(f"Unsupported basis_mode: {self.basis_mode}")

        # Compute mutation matrix
        mutation_logits = torch.einsum("blk,bkij->blij", site_weights, basis_logits)
        if self.selection_model is not None and self.apply_selection:
            seq_embed = self.embed_sequence(x)[:, 1:, :]  # (B, L, D_seq)

            # # Concatenate and project to query space
            # attn_input = torch.cat(
            #     [seq_embed, pos_emb], dim=-1
            # )  # (B, L, D_seq + D_pos)
            # queries = self.attn_query_proj(attn_input)  # (B, L, D)
            selection_logits = self.selection_model(site_context=seq_embed, t=t)  # (B, L, N)
            mutation_logits = mutation_logits + selection_logits.unsqueeze(2)  # Add to each source row
        return mutation_logits

    def reverse_edit_field(self, x, t):
        # seq_with_cls = self.embed_sequence(x)
        time_embedding = self.embed_time(t)
        # time_embedding = time_embedding.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, L, D_pos)

        # Get positional embeddings for each position
        positions = torch.arange(self.sequence_length, device=x.device)  # (L,)
        pos_emb = self.position_embeddings(positions)  # (L, D_pos)
        pos_emb = pos_emb.unsqueeze(0).expand(x.size(0), -1, -1)  # (B, L, D_pos)

        # Compute weights for sites
        if self.routing_mode == "sitewise":
            # Only use position embeddings
            site_weights = self.site_weights_net(pos_emb)
        elif self.routing_mode == "context":
            # Full context-aware embedding
            seq_with_cls = self.embed_sequence(x)
            mlp_input = torch.cat([seq_with_cls[:, 1:, :], pos_emb], dim=-1)
            site_weights = self.site_weights_net(mlp_input)
        elif self.routing_mode == "attention":
            seq_embed = self.embed_sequence(x)[:, 1:, :]  # (B, L, D_seq)
            # Concatenate and project to query space
            attn_input = torch.cat(
                [seq_embed, pos_emb], dim=-1
            )  # (B, L, D_seq + D_pos)
            queries = self.attn_query_proj(attn_input)  # (B, L, D)

            # Compute attention over basis keys
            logits = torch.einsum("bld,kd->blk", queries, self.basis_keys)  # (B, L, K)
            site_weights = logits / self.routing_temp
        else:
            raise ValueError(f"Unsupported routing_mode: {self.routing_mode}")
        site_weights = torch.softmax(site_weights, dim=-1)  # (L, K)

        # Compute basis mutation matrices
        if self.basis_mode == "time_dependent":
            basis_logits = self.basis_generator(time_embedding).view(
                -1, self.num_basis, self.num_residues, self.num_residues
            )
        elif self.basis_mode == "fixed":
            basis_logits = self.basis_logits.unsqueeze(0).expand(x.size(0), -1, -1, -1)
        else:
            raise ValueError(f"Unsupported basis_mode: {self.basis_mode}")

        mutation_logits = torch.einsum("blk,bkij->blij", site_weights, basis_logits)
        if self.selection_model is not None and self.apply_selection:
            seq_embed = self.embed_sequence(x)[:, 1:, :]  # (B, L, D_seq)

            # # Concatenate and project to query space
            # attn_input = torch.cat(
            #     [seq_embed, pos_emb], dim=-1
            # )  # (B, L, D_seq + D_pos)
            # queries = self.attn_query_proj(attn_input)  # (B, L, D)
            selection_logits = self.selection_model(site_context=seq_embed, t=t)  # (B, L, N)
            mutation_logits = mutation_logits + selection_logits.unsqueeze(2)  # Add to each source row
        return mutation_logits

    def apply_edits(self, x, transition_matrix):
        """
        Apply site-wise categorical transition using learned mutation matrix.

        Args:
            x: One-hot sequence (B, L, N)
            transition_matrix: Transition logits (B, L, N, N) — from base j to base k

        Returns:
            x_t: Updated sequence distribution (B, L, N)
        """
        # Apply softmax to get valid transition probabilities over targets
        transition_probs = F.softmax(transition_matrix, dim=-1)  # (B, L, N, N)

        # Matrix multiply one-hot (as source distribution) with transition matrix
        x_t = torch.einsum("blj,bljk->blk", x, transition_probs)
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
        # Get edit field prediction
        edit_field = self.forward_edit_field(x, t)
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
        # Get reverse edit field prediction
        reverse_field = self.reverse_edit_field(x, t)
        x_t_minus_1 = self.apply_edits(x, reverse_field)
        return x_t_minus_1
