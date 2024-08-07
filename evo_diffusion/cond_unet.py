from flax import linen as nn


class DownsampleBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        kernel_size = (3,)
        strides = (2,)

        x = nn.Conv(
            self.features, kernel_size=kernel_size, strides=strides, padding="SAME"
        )(x)
        x = nn.relu(x)
        return x


class UpsampleBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        kernel_size = (3,)
        strides = (2,)

        x = nn.ConvTranspose(
            self.features, kernel_size=kernel_size, strides=strides, padding="SAME"
        )(x)

        x = nn.relu(x)
        return x


class ConditionalUNet(nn.Module):
    conditioning_dim: int
    num_characters: int
    base_features: int = 64
    depth: int = 4

    def setup(self):
        self.down_blocks = [
            DownsampleBlock(self.base_features * (2**i)) for i in range(self.depth)
        ]
        self.up_blocks = [
            UpsampleBlock(self.base_features * (2**i))
            for i in range(self.depth - 1, -1, -1)
        ]
        self.conditioning_embeddings = [
            nn.Dense(self.base_features * (2 ** (self.depth - i - 1)))
            for i in range(self.depth)
        ]

    @nn.compact
    def __call__(self, x, condition):

        # print(self.down_blocks)
        # print(self.up_blocks)
        # Encoder
        connections = []
        for down_block in self.down_blocks:
            x = down_block(x)
            connections.append(x)

        # Bottleneck
        b = nn.Conv(
            self.base_features * (2 ** (self.depth)),
            kernel_size=(3,),
            strides=(2,),
            padding="SAME",
        )(x)
        b = nn.relu(b)

        # Decoder
        for up_block, connection, embedding_layer in zip(
            self.up_blocks, reversed(connections), self.conditioning_embeddings
        ):
            b = up_block(b)
            b = b[:, : connection.shape[1], :]
            b = b + connection

            # Create conditioning embedding for this upsampling step
            condition_embedding = embedding_layer(condition)  # (batch_size, embed_dim)
            condition_embedding = condition_embedding[:, None, :]

            # Add conditioning embedding to the upsampled feature map using broadcasting
            # print(b.shape, condition_embedding.shape)
            b = b + condition_embedding

        # Output layer to project to num_characters
        output = nn.Conv(self.num_characters, kernel_size=(3,), padding="SAME")(b)

        return output
