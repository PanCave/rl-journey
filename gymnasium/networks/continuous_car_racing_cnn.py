import torch.nn as nn
import torch

class ContinuousCarRacingPolicy(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], action_dim: int):
        super(ContinuousCarRacingPolicy, self).__init__()
        # CNN layers to process the image
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[2], 16, kernel_size=8, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calculate the size of flattened features
        conv_output_size = self._get_conv_output(input_shape)

        # Fully connected layers (3 hidden layers with 128 neurons each)
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim * 2)
        )

    def _get_conv_output(self, shape) -> int:
        # Forward pass with dummy input to get output shape
        bs = 1
        dummy_data = torch.zeros(bs, shape[2], shape[0], shape[1])
        x = self.conv_layers(dummy_data)

        return x.flatten(1).size(1)

    def forward(self, tensor_input: torch.Tensor) -> torch.Tensor:
        # Ensure input has the right format (batch_size, channels, height, width)
        # Original shape: (batch_size, height, width, channels)
        # No longer needed, as we now permute in preprocessing
        #tensor_input = tensor_input.permute(0, 3, 1, 2)

        # Normalize pixel values to [0, 1]
        tensor_input = tensor_input.float() / 255.0

        # CNN layers
        tensor_input = self.conv_layers(tensor_input)

        # Flatten
        tensor_input = tensor_input.flatten(1)

        # Fully connected layers
        tensor_input = self.fc_layers(tensor_input)

        return tensor_input
    
    
class ContinuousCarRacingCritic(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], action_dim: int):
        super(ContinuousCarRacingCritic, self).__init__()
        # CNN layers to process the image
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[2], 16, kernel_size=8, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calculate the size of flattened features
        conv_output_size = self._get_conv_output(input_shape)

        # Fully connected layers (3 hidden layers with 128 neurons each)
        self.fc_layers_1 = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )
        self.fc_layers_2 = nn.Sequential(
            nn.Linear(64 + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def _get_conv_output(self, shape) -> int:
        # Forward pass with dummy input to get output shape
        bs = 1
        dummy_data = torch.zeros(bs, shape[2], shape[0], shape[1])
        x = self.conv_layers(dummy_data)

        return x.flatten(1).size(1)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        # Ensure input has the right format (batch_size, channels, height, width)
        # Original shape: (batch_size, height, width, channels)
        # No longer needed, as we now permute in preprocessing
        #tensor_input = tensor_input.permute(0, 3, 1, 2)

        # Normalize pixel values to [0, 1]
        state = state.float() / 255.0

        # CNN layers
        conv_output = self.conv_layers(state)

        # Flatten
        flattened_conv_output = conv_output.flatten(1)
        
        # Fully connected layers
        fc1_output = self.fc_layers_1(flattened_conv_output)
        stacked_fc2_input = torch.vstack([fc1_output, action])
        fc2_output  = self.fc_layers_2(stacked_fc2_input)

        return fc2_output