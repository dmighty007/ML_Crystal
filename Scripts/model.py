# Standard library
import numpy as np
from typing import List, Callable

# Pytorch library
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# For progress bars
from tqdm import tqdm

class BetaVAE(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 code_dim: int,
                 activation_fn: Callable = nn.GELU,
                 output_activation_fn: Callable = None):
        """
        Improved Beta-Variational Autoencoder (Beta-VAE) Model.

        Args:
            input_dim (int): Dimension of the input data.
            hidden_dims (List[int]): List of hidden layer dimensions for encoder/decoder.
            code_dim (int): Dimension of the latent space.
            activation_fn (Callable): Activation function for hidden layers.
            output_activation_fn (Callable, optional): Activation for the final decoder layer.
                                                       Defaults to None (linear output).
                                                       Use nn.Sigmoid for data in [0, 1].
        """
        super().__init__()
        self.code_dim = code_dim

        # --- Build Encoder ---
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, h_dim), activation_fn()])
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        self.mu = nn.Linear(hidden_dims[-1], code_dim)
        self.logvar = nn.Linear(hidden_dims[-1], code_dim)

        # --- Build Decoder ---
        decoder_layers = []
        in_dim = code_dim
        # Reverse the hidden_dims for the decoder
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(in_dim, h_dim), activation_fn()])
            in_dim = h_dim

        # Final layer to reconstruct the input
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        if output_activation_fn:
            decoder_layers.append(output_activation_fn())

        self.decoder = nn.Sequential(*decoder_layers)

        self._init_weights()

    def _init_weights(self):
        """Initializes weights using Xavier Uniform for better training stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Encodes input to latent space distribution."""
        encoded_features = self.encoder(x)
        return self.mu(encoded_features), self.logvar(encoded_features)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from N(mu, var)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent vector back to the input space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """Full forward pass of the VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, logvar

    @staticmethod
    def loss_function(reconstructed_x, x, mu, logvar, beta: float) -> dict:
        """
        Calculates Beta-VAE loss.

        Args:
            reconstructed_x: The model's output.
            x: The original input.
            mu: The latent mean.
            logvar: The latent log-variance.
            beta (float): The weight for the KL divergence term.

        Returns:
            dict: A dictionary with total loss, reconstruction loss, and KL divergence.
                  Note: Losses are averaged over the batch.
        """
        # Reconstruction Loss (e.g., Mean Squared Error).
        # For binary data (e.g., MNIST), nn.functional.binary_cross_entropy_with_logits is better.
        recon_loss = nn.functional.mse_loss(reconstructed_x, x, reduction='mean')

        # KL Divergence
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + beta * kl_div

        return {'loss': total_loss, 'recon_loss': recon_loss, 'kl_div': kl_div}

# ---------------------------------------------------------------------------- #
#                          Decoupled Training Function                         #
# ---------------------------------------------------------------------------- #

def train_model(model: BetaVAE,
                train_loader: DataLoader,
                val_loader: DataLoader,
                optimizer: optim.Optimizer,
                epochs: int,
                beta: float,
                device: torch.device,
                patience: int = 10,
                scheduler: optim.lr_scheduler._LRScheduler = None):
    """
    Trains the BetaVAE model with early stopping and learning rate scheduling.
    """
    print(f"Training on {device.type.upper()} with Beta = {beta}")

    best_val_loss = float('inf')
    stop_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'kl_div': [], 'recon_loss': []}

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0

        # Use tqdm for a progress bar over batches
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for (x_batch,) in train_pbar:
            x_batch = x_batch.to(device)

            reconstructed_x, mu, logvar = model(x_batch)
            loss_dict = model.loss_function(reconstructed_x, x_batch, mu, logvar, beta)
            loss = loss_dict['loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0
        total_kl_div = 0
        total_recon_loss = 0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
        with torch.no_grad():
            for (x_batch,) in val_pbar:
                x_batch = x_batch.to(device)
                reconstructed_x, mu, logvar = model(x_batch)
                loss_dict = model.loss_function(reconstructed_x, x_batch, mu, logvar, beta)

                total_val_loss += loss_dict['loss'].item()
                total_kl_div += loss_dict['kl_div'].item()
                total_recon_loss += loss_dict['recon_loss'].item()
                val_pbar.set_postfix({'loss': f"{loss_dict['loss'].item():.4f}"})

        avg_val_loss = total_val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        history['kl_div'].append(total_kl_div / len(val_loader))
        history['recon_loss'].append(total_recon_loss / len(val_loader))

        print(f'Epoch: {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}')

        # --- Learning Rate Scheduler Step ---
        if scheduler:
            scheduler.step(avg_val_loss)

        # --- Early Stopping Check ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            stop_counter = 0
            # torch.save(model.state_dict(), 'best_beta_vae_model.pt')
        else:
            stop_counter += 1
            if stop_counter >= patience:
                print(f'Early stopping triggered after {patience} epochs without improvement.')
                break

    print("Training finished.")
    return history


# ---------------------------------------------------------------------------- #
#                                Example Usage                                 #
# ---------------------------------------------------------------------------- #

if __name__ == '__main__':
    # --- 1. Setup Environment and Data ---
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    INPUT_DIM = 784  # e.g., flattened 28x28 images

    # Generate dummy data
    train_data = np.random.rand(1000, INPUT_DIM).astype(np.float32)
    val_data = np.random.rand(200, INPUT_DIM).astype(np.float32)

    # --- 2. Define Hyperparameters ---
    HIDDEN_DIMS = [256, 128, 64]
    CODE_DIM = 10
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    BETA = 1.0  # beta=1.0 is a standard VAE. Increase for more disentanglement.
    PATIENCE = 10

    # --- 3. Create DataLoaders ---
    train_dataset = TensorDataset(torch.from_numpy(train_data))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = TensorDataset(torch.from_numpy(val_data))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 4. Instantiate Model, Optimizer, and Scheduler ---
    model = BetaVAE(input_dim=INPUT_DIM,
                    hidden_dims=HIDDEN_DIMS,
                    code_dim=CODE_DIM,
                    # For data in [0,1], nn.Sigmoid might be a good output activation
                    # output_activation_fn=nn.Sigmoid
                   ).to(DEVICE)

    print("Model Architecture:")
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Scheduler reduces LR on validation loss plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)

    # --- 5. Train the Model ---
    history = train_model(model=model,
                          train_loader=train_loader,
                          val_loader=val_loader,
                          optimizer=optimizer,
                          epochs=EPOCHS,
                          beta=BETA,
                          device=DEVICE,
                          patience=PATIENCE,
                          scheduler=scheduler)

    # --- 6. Inspect Training History ---
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
