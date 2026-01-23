from typing import Optional, List
import numpy as np
from copy import deepcopy

# --- Re-using imports and base classes from the original code ---
from base import Transformer, Model, EstimatorTransformer
from base_torch import DLEstimatorMixin
from util import map_data
from deeptime.util.platform import try_import
from deeptime.util.torch import MLP

torch = try_import("torch")

# Helper functions and VAEEncoder/VAEModel classes remain the same...
def _reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu

class VAEEncoder(MLP):
    def __init__(self, units: List[int], nonlinearity=None):
        import torch.nn as nn
        if nonlinearity is None:
            nonlinearity = nn.ELU
        super().__init__(units[:-1], nonlinearity=nonlinearity, initial_batchnorm=False,
                         output_nonlinearity=nonlinearity)
        latent_input_dim = units[-2]
        latent_output_dim = units[-1]
        self._to_mu = nn.Linear(latent_input_dim, latent_output_dim)
        self._to_logvar = nn.Linear(latent_input_dim, latent_output_dim)

    def forward(self, inputs):
        out = self._sequential(inputs)
        return self._to_mu(out), self._to_logvar(out)


class VAEModel(Model, Transformer):
    def __init__(self, encoder, decoder, device=None, dtype=np.float32):
        self._encoder = encoder
        self._decoder = decoder
        self._device = device
        self._dtype = dtype

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    def _encode(self, x: "torch.Tensor"):
        mu, logvar = self._encoder(x)
        return _reparameterize(mu, logvar)

    def get_latent_parameters(self, data, **kwargs):
        out_mu, out_logvar = [], []
        for data_tensor in map_data(data, device=self._device, dtype=self._dtype):
            mu, logvar = self._encoder(data_tensor)
            out_mu.append(mu.cpu().numpy())
            out_logvar.append(logvar.cpu().numpy())
        if len(out_mu) == 1:
            return out_mu[0], out_logvar[0]
        return out_mu, out_logvar

    def transform(self, data, **kwargs):
        out = []
        for data_tensor in map_data(data, device=self._device, dtype=self._dtype):
            out.append(self._encode(data_tensor).cpu().numpy())
        return out if len(out) > 1 else out[0]


class VAE(EstimatorTransformer, DLEstimatorMixin):
    r""" Variational Autoencoder (VAE) with Early Stopping and LR Scheduling.

    Parameters
    ----------
    encoder : torch.nn.Module
        Encoder module, typically a `VAEEncoder`.
    decoder : torch.nn.Module
        Decoder module.
    beta : float, default=1.0
        The weight of the KL divergence term in the loss function.
    optimizer : str or callable, default='Adam'
        The optimizer to use.
    learning_rate : float, default=5e-4
        The initial learning rate for the optimizer.
    device : str, default='cpu'
        The device to train on.
    early_stopping_patience : int, default=10
        Number of epochs to wait for improvement in validation loss before stopping.
    early_stopping_min_delta : float, default=1e-4
        Minimum change in validation loss to be considered an improvement.
    use_lr_scheduler : bool, default=True
        Whether to use a ReduceLROnPlateau learning rate scheduler.
    """
    _MUTABLE_INPUT_DATA = True

    def __init__(self, encoder: "torch.nn.Module", decoder: "torch.nn.Module", beta: float = 1.0,
                 optimizer='Adam', learning_rate: float = 5e-4, device='cpu',
                 # NEW: Early stopping and scheduler parameters
                 early_stopping_patience: int = 10,
                 early_stopping_min_delta: float = 1e-4,
                 use_lr_scheduler: bool = True):
        import torch.nn as nn
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        super().__init__()
        self.device = device
        self._encoder = encoder.to(self.device)
        self._decoder = decoder.to(self.device)
        self._beta = beta
        self.learning_rate = learning_rate

        # Setup optimizer
        self.setup_optimizer(optimizer, list(encoder.parameters()) + list(decoder.parameters()))

        # NEW: Setup optional learning rate scheduler
        self.use_lr_scheduler = use_lr_scheduler
        if self.use_lr_scheduler:
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=early_stopping_patience // 2, factor=0.5)

        self._mse_loss = nn.MSELoss(reduction='sum')
        self._train_losses = []
        self._val_losses = []

        self._es_patience = early_stopping_patience
        self._es_min_delta = early_stopping_min_delta

    def forward(self, x: "torch.Tensor"):
        mu, logvar = self._encoder(x)
        z = _reparameterize(mu, logvar)
        x_hat = self._decoder(z)
        return x_hat, mu, logvar

    def loss_function(self, reconstructed_x, x, mu, logvar, beta: float) -> dict:
        """
        Calculates the VAE loss, which is a sum of reconstruction loss and KL divergence.
        """
        reconstruction_loss = self._mse_loss(reconstructed_x, x)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = reconstruction_loss + beta * kld_loss
        return {'loss': total_loss, 'recon_loss': reconstruction_loss, 'kld_loss': kld_loss}

    @property
    def train_losses(self) -> np.ndarray:
        return np.array(self._train_losses)

    @property
    def validation_losses(self) -> np.ndarray:
        return np.array(self._val_losses)

    def fit(self, data_loader: "torch.utils.data.DataLoader", n_epochs: int = 100,
            validation_loader: Optional["torch.utils.data.DataLoader"] = None, **kwargs):
        r""" Fits the VAE model with early stopping. """

        if validation_loader is None:
            print("Warning: No validation_loader provided. Early stopping and LR scheduling are disabled.")

        step = 0
        patience_counter = 0
        best_val_loss = float('inf')
        best_encoder_state = None
        best_decoder_state = None

        for epoch in range(n_epochs):
            self._encoder.train()
            self._decoder.train()

            # --- Training Loop ---
            for batch in data_loader:
                batch = batch.to(device=self.device)
                self.optimizer.zero_grad()

                # Forward pass
                reconstructed_x, mu, logvar = self.forward(batch)

                # Calculate loss
                loss_dict = self.loss_function(reconstructed_x, batch, mu, logvar, self._beta)
                loss_value = loss_dict['loss']

                loss_value.backward()
                self.optimizer.step()
                self._train_losses.append((step, loss_value.item()))
                step += 1

            # --- Validation and Early Stopping Logic ---
            if validation_loader is not None:
                self._encoder.eval()
                self._decoder.eval()

                with torch.no_grad():
                    lval = []
                    for batch in validation_loader:
                        batch = batch.to(device=self.device)
                        reconstructed_x, mu, logvar = self.forward(batch)
                        loss_dict = self.loss_function(reconstructed_x, batch, mu, logvar, self._beta)
                        loss_value = loss_dict['loss'].item()
                        lval.append(loss_value)

                # Calculate the average validation loss for the epoch
                current_val_loss = np.mean(lval)
                self._val_losses.append((step, current_val_loss))

                if self.use_lr_scheduler:
                    self.scheduler.step(current_val_loss)

                if current_val_loss < best_val_loss - self._es_min_delta:
                    # Improvement found, save model and reset counter
                    best_val_loss = current_val_loss
                    patience_counter = 0
                    best_encoder_state = deepcopy(self._encoder.state_dict())
                    best_decoder_state = deepcopy(self._decoder.state_dict())
                    print(f"Epoch {epoch+1}/{n_epochs}: Val loss improved to {best_val_loss:.4f}. Saving model.")
                else:
                    # No improvement, increment counter
                    patience_counter += 1
                    print(f"Epoch {epoch+1}/{n_epochs}: Val loss did not improve from {best_val_loss:.4f}. Patience: {patience_counter}/{self._es_patience}")

                if patience_counter >= self._es_patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
                    break # Exit the training loop

        if best_encoder_state is not None:
            print(f"Restoring model to best validation loss of {best_val_loss:.4f}.")
            self._encoder.load_state_dict(best_encoder_state)
            self._decoder.load_state_dict(best_decoder_state)

        return self

    def fetch_model(self) -> VAEModel:
        return VAEModel(deepcopy(self._encoder), deepcopy(self._decoder),
                        device=self.device, dtype=self.dtype)
