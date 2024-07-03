import torch
torch.set_float32_matmul_precision('high')
import wandb
import torch.nn as nn


class MLPSmall(nn.Module):
    """
        todo
    """

    def __init__(self, config):
        super().__init__()
        _ = self.check_signature(config)

        self.input_dim = config['input_dim']
        self.learning_rate = config['learning_rate']
        self.seed = config['seed']
        self.model_name = config['model_name']
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.classifier = nn.Sequential(
            nn.Linear(in_features=config['input_dim'], out_features=10, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=10, out_features=1, bias=True))

    def forward(self, x):
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        self.train()
        z, s, t = batch
        t_pred = self(z.to(self.device))
        return self.criterion(t.to(self.device), t_pred)

    def validation_step(self, batch, batch_idx):
        self.eval()
        z, s, t = batch
        t_pred = self(z.to(self.device))
        return self.criterion(t.to(self.device), t_pred)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @classmethod
    def check_signature(cls, config):
        """
        Class method to check if the provided configuration dictionary is valid.

        Args:
            config (dict): A dictionary containing configuration parameters for the model.

        Returns:
            bool: True if the configuration is valid; False otherwise.

        Raises:
            ValueError: If the configuration is missing required keys or has incorrect data types.

        Example:
            config = {
                'input_dim': 2,
                'learning_rate': 1e-3,
                'model_name': 'mlp_small',
                'seed': 42
            }
        """

        # Define the required keys and their expected types
        required_keys = [
            ('input_dim', int),
            ('learning_rate', float),
            ('model_name', str),
            ('seed', int),
        ]

        # Check if all required keys are present and have the correct data types
        for key, expected_type in required_keys:
            if key not in config:
                raise ValueError(f"Missing '{key}' in the configuration.")

            if not isinstance(config[key], expected_type):
                raise ValueError(f"Invalid type for '{key}'. Expected {expected_type}, got {type(config[key])}.")
        return True
