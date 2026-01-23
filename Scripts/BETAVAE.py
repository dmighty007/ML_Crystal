# Standard library
import numpy as np
from typing import List

# Pytorch library
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class BetaVAE(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 code_dim: int):
        """
        Beta-Variational Autoencoder (Beta-VAE) Model.

        Args:
            input_dim (int): The dimension of the input data (e.g., flattened image size).
            hidden_dims (List[int]): A list of integers specifying the number of neurons in each hidden layer of the encoder and decoder.
            code_dim (int): The dimension of the latent space (the "code").
        """
        super(BetaVAE, self).__init__()

        self.input_dim = input_dim
        self.code_dim = code_dim

        # --- Build Encoder ---
        encoder_layers = []
        # The first layer connects the input to the first hidden dimension.
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.ReLU()) # Using ReLU activation
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Layers to get mean and log-variance from the encoder's output
        self.mu = nn.Linear(hidden_dims[-1], code_dim)
        self.logvar = nn.Linear(hidden_dims[-1], code_dim)

        # --- Build Decoder ---
        decoder_layers = []
        # The decoder layers are the reverse of the encoder layers.
        reversed_hidden_dims = list(reversed(hidden_dims))
        in_dim = code_dim
        for i, h_dim in enumerate(reversed_hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            in_dim = h_dim

        # The final layer maps back to the original input dimension.
        decoder_layers.append(nn.Linear(reversed_hidden_dims[-1], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initializes the weights of the linear layers using Xavier Uniform initialization.
        This helps in preventing vanishing or exploding gradients.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to allow backpropagation through a random node.

        Args:
            mu (torch.Tensor): The mean of the latent Gaussian distribution.
            logvar (torch.Tensor): The log-variance of the latent Gaussian distribution.

        Returns:
            torch.Tensor: A sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # Sample from a standard normal distribution
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Encodes the input by passing it through the encoder network
        and returns the latent mean and log-variance.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor, torch.Tensor): The latent mean and log-variance.
        """
        encoded_features = self.encoder(x)
        mu = self.mu(encoded_features)
        logvar = self.logvar(encoded_features)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent vector by passing it through the decoder network.

        Args:
            z (torch.Tensor): The latent vector.

        Returns:
            torch.Tensor: The reconstructed output.
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        The forward pass of the VAE.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): The reconstructed output, latent mean, and latent log-variance.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, logvar

    def loss_function(self, reconstructed_x, x, mu, logvar, beta: float) -> dict:
        """
        Calculates the VAE loss, which is a sum of reconstruction loss and KL divergence.

        Args:
            reconstructed_x: The model's output.
            x: The original input.
            mu: The latent mean.
            logvar: The latent log-variance.
            beta (float): The weight for the KL divergence term.

        Returns:
            dict: A dictionary containing the total loss, reconstruction loss, and KL divergence.
        """
        # Using Mean Squared Error for reconstruction loss, summed over features
        recon_loss = nn.functional.mse_loss(reconstructed_x, x, reduction='sum')

        # KL Divergence between the learned distribution and a standard normal distribution
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss is the sum of reconstruction loss and the weighted KL divergence
        total_loss = recon_loss + beta * kl_div

        return {'loss': total_loss, 'recon_loss': recon_loss, 'kl_div': kl_div}


    def fit(self,
            train_data: np.ndarray,
            val_data: np.ndarray,
            epochs: int = 50,
            batch_size: int = 64,
            learning_rate: float = 1e-4,
            beta: float = 1.0,
            patience: int = 5,
            shuffle: bool = True):
        """
        Trains the VAE model.

        Args:
            train_data (np.ndarray): The training dataset.
            val_data (np.ndarray): The validation dataset.
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
            learning_rate (float): The learning rate for the Adam optimizer.
            beta (float): The weight for the KL divergence term.
            patience (int): Number of epochs to wait for improvement before early stopping.
            shuffle (bool): Whether to shuffle the data loaders.
        """
        # Determine device (GPU or CPU)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)
        print(f"Training on {device.upper()}")

        # Create DataLoaders
        train_dataset = TensorDataset(torch.Tensor(train_data))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

        val_dataset = TensorDataset(torch.Tensor(val_data))
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Setup optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Early stopping variables
        best_val_loss = float('inf')
        stop_counter = 0

        # Training history
        self.history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            # --- Training Phase ---
            self.train()
            total_train_loss = 0
            for (x_batch,) in train_dataloader:
                x_batch = x_batch.to(device)

                # Forward pass
                reconstructed_x, mu, logvar = self(x_batch)

                # Calculate loss
                loss_dict = self.loss_function(reconstructed_x, x_batch, mu, logvar, beta)
                loss = loss_dict['loss']

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_dataset)
            self.history['train_loss'].append(avg_train_loss)

            # --- Validation Phase ---
            self.eval()
            total_val_loss = 0
            with torch.no_grad():
                for (x_batch,) in val_dataloader:
                    x_batch = x_batch.to(device)
                    reconstructed_x, mu, logvar = self(x_batch)
                    loss_dict = self.loss_function(reconstructed_x, x_batch, mu, logvar, beta)
                    total_val_loss += loss_dict['loss'].item()

            avg_val_loss = total_val_loss / len(val_dataset)
            self.history['val_loss'].append(avg_val_loss)

            print(f'Epoch: {epoch + 1}/{epochs} | '
                  f'Train Loss: {avg_train_loss:.5f} | '
                  f'Val Loss: {avg_val_loss:.5f}')

            # --- Early Stopping Check ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                stop_counter = 0
                # Optionally save the best model
                # torch.save(self.state_dict(), 'best_model.pt')
            else:
                stop_counter += 1
                if stop_counter >= patience:
                    print(f'Early stopping triggered after {patience} epochs without improvement.')
                    break
        print("Training finished.")

# --- Example Usage ---
if __name__ == '__main__':
    # --- 1. Generate Dummy Data ---
    # Imagine we have 1000 data points, each being a flattened 28x28 image (784 features)
    INPUT_DIM = 784
    train_data = np.random.rand(1000, INPUT_DIM).astype(np.float32)
    val_data = np.random.rand(200, INPUT_DIM).astype(np.float32)

    # --- 2. Define Model Architecture ---
    # We define the hidden layers and the size of the latent code
    HIDDEN_DIMS = [256, 128, 64]
    CODE_DIM = 10

    # --- 3. Instantiate the Model ---
    model = BetaVAE(input_dim=INPUT_DIM,
                    hidden_dims=HIDDEN_DIMS,
                    code_dim=CODE_DIM)
    print("Model Architecture:")
    print(model)

    # --- 4. Train the Model ---
    # We can now train the model by calling the fit method
    model.fit(train_data=train_data,
              val_data=val_data,
              epochs=100,
              batch_size=32,
              learning_rate=0.001,
              beta=1.0,  # beta=1.0 is a standard VAE. Increase for more disentanglement.
              patience=10)

    # --- 5. Inspect Training History ---
    # The training history is stored in the model object
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(model.history['train_loss'], label='Train Loss')
    plt.plot(model.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
