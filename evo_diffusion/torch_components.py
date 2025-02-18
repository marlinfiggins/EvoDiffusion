import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_t=100):
        """
        Sinusoidal positional embedding for time.

        Args:
            embedding_dim: Dimensionality of the embeddings.
            max_t: Maximum time step for embeddings.
        """
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_t = max_t

    def forward(self, t):
        """
        Generate sinusoidal embeddings.

        Args:
            t: Time step tensor of shape (batch_size, 1).

        Returns:
            embeddings: Sinusoidal embeddings of shape (batch_size, embedding_dim).
        """
        device = t.device
        half_dim = self.embedding_dim // 2
        div_term = torch.exp(
            torch.arange(half_dim, device=device) * -(torch.log(torch.tensor(self.max_t)) / half_dim)
        )
        pos = t[:, None] * div_term
        embeddings = torch.cat([torch.sin(pos), torch.cos(pos)], dim=-1)
        if self.embedding_dim % 2 != 0:  # Handle odd embedding sizes
            embeddings = torch.cat([embeddings, torch.zeros_like(t)], dim=-1)
        return embeddings

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, mlp_params):
        """
        A flexible MLP that constructs layers based on mlp_params.

        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output features.
            mlp_params (dict): Dictionary containing parameters for the MLP.
                Example:
                {
                    "feedforward_dim": 128,     # List of hidden layer dimensions
                    "activation": "ReLU",       # Activation function
                    "dropout": 0.1              # Dropout rate (optional)
                }
        """
        super(MLP, self).__init__()

        layers = []
        prev_dim = input_dim

        # Add hidden layers
        for hidden_dim in mlp_params.get("feedforward_dim", []):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            activation = getattr(nn, mlp_params.get("activation", "ReLU"))()  # Default to ReLU
            layers.append(activation)
            if mlp_params.get("dropout", 0) > 0:
                layers.append(nn.Dropout(mlp_params["dropout"]))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class TransformerSequenceEmbedding(nn.Module):
    def __init__(self, sequence_length, embedding_dim, num_heads, num_layers, feedforward_dim):
        """
        Transformer-based sequence embedding with mean pooling for full-sequence embedding.

        Args:
            sequence_length: Length of the input sequence.
            embedding_dim: Dimensionality of the embeddings.
            num_heads: Number of attention heads.
            num_layers: Number of transformer layers.
            feedforward_dim: Dimension of the feedforward network.
        """
        super(TransformerSequenceEmbedding, self).__init__()

        # Project residue dimension (4 for one-hot) to embedding dimension
        self.residue_projection = nn.Linear(4, embedding_dim)

        # Positional embedding for sequence positions
        self.positional_embedding = nn.Embedding(sequence_length, embedding_dim)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # Final layer normalization (optional but improves stability)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        """
        Apply transformer layers to the input sequence and return a pooled embedding.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, residue_dim).

        Returns:
            embedding: Embedded sequence tensor of shape (batch_size, embedding_dim).
        """
        # Project residues to embedding space
        residue_embeddings = self.residue_projection(x)  # Shape: (batch_size, sequence_length, embedding_dim)

        # Generate positional embeddings
        batch_size, sequence_length, _ = x.shape
        positions = torch.arange(sequence_length, device=x.device).unsqueeze(0).expand(batch_size, sequence_length)
        positional_embeddings = self.positional_embedding(positions)  # Shape: (batch_size, sequence_length, embedding_dim)

        # Combine residue and positional embeddings
        x = residue_embeddings + positional_embeddings  # Shape: (batch_size, sequence_length, embedding_dim)

        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Layer normalization
        x = self.layer_norm(x)

        # Pool over the sequence length to get a single embedding per sequence
        #sequence_embedding = x.mean(dim=1)  # Shape: (batch_size, embedding_dim)
        sequence_embedding = x
        return sequence_embedding

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = F.max_pool2d(x, kernel_size=(2, 1))
        return self.double_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=(2, 1), stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        out = torch.cat([skip, x], dim=1)
        return self.conv(out)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, input_channels, output_channels, features, bilinear=False):
        """
        Args:
            input_channels: Number of input channels.
            output_channels: Number of output channels.
            features: List of channel sizes for each downsampling/up sampling layer.
            bilinear: Whether to use bilinear upsampling.
        """
        super(UNet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.bilinear = bilinear

        # Downsampling path
        self.downs = nn.ModuleList()
        for i in range(len(features) - 1):
            self.downs.append(Down(features[i], features[i + 1]))

        # Bottleneck layer
        factor = 2 if bilinear else 1
        self.bottleneck = DoubleConv(features[-1], features[-1] // factor)

        # Upsampling path
        self.ups = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.ups.append(Up(features[i], features[i - 1] // factor, bilinear))

        # Final 1x1 convolution
        self.outc = OutConv(features[0], self.output_channels)

    def forward(self, x):
        skip_connections = [x]

        # Downsampling
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)

        skip_connections = skip_connections[::-1]

        # Bottleneck
        x = self.bottleneck(x)

        # Upsampling
        for idx, up in enumerate(self.ups):
            x = up(x, skip_connections[idx+1])
        return self.outc(x)