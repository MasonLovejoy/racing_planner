from pathlib import Path

import torch
import torch.nn as nn

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 4,
    ):
        """
        Multi-layer perceptron planner for autonomous racing.
        
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
            hidden_dim (int): dimension of hidden layers
            num_layers (int): number of hidden layers
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        input_dim = n_track * 2 * 2
        output_dim = n_waypoints * 2

        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))

        for layer_idx in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """

        batch_size = track_left.shape[0]
        track_left_flat = track_left.reshape(batch_size, -1)
        track_right_flat = track_right.reshape(batch_size, -1)

        x = torch.cat([track_left_flat, track_right_flat], dim=1)
        output = self.mlp(x)
        waypoints = output.reshape(batch_size, self.n_waypoints, 2)

        return waypoints


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 128,
        n_head: int = 8,
        dropout: float = 0.1,
        num_layers: int = 6,
    ):
        """
        Transformer-based planner for autonomous racing trajectory prediction.
        
        Args:
            n_track (int): number of track boundary points
            n_waypoints (int): number of waypoints to predict
            d_model (int): transformer embedding dimension
            n_head (int): number of attention heads
            dropout (float): dropout probability
            num_layers (int): number of transformer decoder layers
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        self.query_embed = nn.Embedding(n_waypoints, d_model)

        self.track_encoder = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.track_embed = nn.Parameter(torch.randn(n_track * 2, d_model))

        decode_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=n_head, 
            dim_feedforward=d_model*4, 
            dropout=dropout, 
            activation='gelu', 
            batch_first=True, 
            norm_first=True
        )

        self.transformer_decoder = nn.TransformerDecoder(decode_layer, num_layers=num_layers)

        self.output = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Linear(d_model//2, 2),
        )


    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        
        batch_size = track_left.shape[0]

        track = torch.cat([track_left, track_right], dim=1)
        track_encoded = self.track_encoder(track)
        track_embedded = track_encoded + self.track_embed.unsqueeze(0)

        query_embedded = self.query_embed.weight
        queries = query_embedded.unsqueeze(0).expand(batch_size, -1, -1)

        transformer_output = self.transformer_decoder(
            tgt=queries,
            memory=track_embedded,
        )

        transformer_output = torch.nn.functional.normalize(transformer_output, dim=-1) * (self.d_model ** 0.5)

        waypoints = self.output(transformer_output)

        return waypoints


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
        base_channels: int = 32,
        dropout: float = 0.20,
    ):
        """
        Vision-based planner using convolutional neural networks.
        
        Args:
            n_waypoints (int): number of waypoints to predict
            base_channels (int): base number of channels in CNN
            dropout (float): dropout probability
        """
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        self.initial = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(base_channels, base_channels*2, num_blocks=2)
        self.layer2 = self._make_layer(base_channels*2, base_channels*4, num_blocks=2)
        self.layer3 = self._make_layer(base_channels*4, base_channels*8, num_blocks=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels*8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_waypoints * 2),
        )

    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=2))

        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Predicts waypoints from raw RGB images.
        
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.fc(x)
        
        waypoints = x.view(-1, self.n_waypoints, 2)
        return waypoints
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.nn.functional.relu(out)
        return out


MODEL_REGISTRY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Loads a planner model by name with optional pre-trained weights.
    
    Args:
        model_name (str): name of the model architecture
        with_weights (bool): whether to load pre-trained weights
        **model_kwargs: additional model configuration parameters
        
    Returns:
        torch.nn.Module: initialized model
    """
    m = MODEL_REGISTRY[model_name](**model_kwargs)

    if with_weights:
        model_path = PROJECT_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # validate model size for deployment
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Saves model state dict to disk.
    
    Args:
        model (torch.nn.Module): model to save
        
    Returns:
        str: path to saved model
    """
    model_name = None

    for n, m in MODEL_REGISTRY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = PROJECT_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Estimates model size in megabytes.
    
    Args:
        model (torch.nn.Module): model to measure
        
    Returns:
        float: estimated size in MB
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024