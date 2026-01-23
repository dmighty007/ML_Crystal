from typing import Optional

import numpy as np
from base import EstimatorTransformer, Model, Transformer
from base_torch import DLEstimatorMixin
from deeptime.util.platform import try_import

# from util import try_import
from util import map_data

#import torch
torch = try_import("torch")

class AutoencoderModel(Model, Transformer):
    r""" Model produced by an autoencoder. Contains the encoder, decoder, and can transform data to
    the latent code.

    Parameters
    ----------
    encoder : torch.nn.Module
        The encoder module.
    decoder : torch.nn.Module
        The decoder module.
    device : torch device or None, default=None
        The device to use.
    dtype : numpy datatype, default=np.float32
        The dtype that is used for transformation of data.

    See Also
    --------
    Autoencoder
    """

    def __init__(self, encoder, decoder, device=None, dtype=np.float32):
        self._encoder = encoder
        self._decoder = decoder
        self._device = device
        self._dtype = dtype

    @property
    def encoder(self):
        r""" The encoder.

        :type: torch.nn.Module
        """
        return self._encoder

    @property
    def decoder(self):
        r""" The decoder.

        :type: torch.nn.Module
        """
        return self._decoder

    def _encode(self, x: "torch.Tensor"):
        return self._encoder(x)

    def transform(self, data, **kwargs):
        r""" Transforms a trajectory (or a list of trajectories) by passing them through the encoder network.

        Parameters
        ----------
        data : array_like or list of array_like
            The trajectory data.
        **kwargs
            Ignored.

        Returns
        -------
        latent_code : ndarray or list of ndarray
            The trajectory / trajectories encoded to the latent representation.
        """
        out = []
        for data_tensor in map_data(data, device=self._device, dtype=self._dtype):
            out.append(self._encode(data_tensor).cpu().numpy())
        return out if len(out) > 1 else out[0]


class Autoencoder(EstimatorTransformer, DLEstimatorMixin):
    r""" Standard autoencoder.

    Parameters
    ----------
    encoder : torch.nn.Module
        Encoder module.
    decoder : torch.nn.Module
        Decoder module, its input features should be compatible with the encoder's output features.
    optimizer : str or callable, default='Adam'
        The optimizer to use, defaults to Adam. If given as string, can be one of 'Adam', 'SGD', 'RMSProp'.
        In case of a callable, the callable should take a `params` parameter list and a `lr` learning rate, yielding
        an optimizer instance based on that.
    learning_rate : float, default=3e-4
        The learning rate that is used for the optimizer.
    """
    _MUTABLE_INPUT_DATA = True

    def __init__(self, encoder: "torch.nn.Module", decoder: "torch.nn.Module",
                 optimizer='Adam', learning_rate=3e-4, device='cpu'):
        import torch.nn as nn

        super().__init__()
        self.device = device
        self._encoder = encoder.to(self.device)
        self._decoder = decoder.to(self.device)
        self.learning_rate = learning_rate
        self.setup_optimizer(optimizer, list(encoder.parameters()) + list(decoder.parameters()))
        self._mse_loss = nn.MSELoss(reduction='sum')
        self._train_losses = []
        self._val_losses = []

    def evaluate_loss(self, x: "torch.Tensor"):
        r""" Evaluates the loss based on input tensors.

        Parameters
        ----------
        x : torch.Tensor
            The tensor that is passed through encoder and decoder networks and used as the reconstruction target.

        Returns
        -------
        loss : torch.Tensor
            The loss.
        """
        return self._mse_loss(x, self._decoder(self._encoder(x)))

    @property
    def train_losses(self) -> np.ndarray:
        r""" The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._train_losses)

    @property
    def validation_losses(self) -> np.ndarray:
        r""" The collected validation scores. First dimension contains the step, second dimension the score.
        Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._val_losses)

    def fit(self, data_loader: "torch.utils.data.DataLoader", n_epochs: int = 5,
            validation_loader: Optional["torch.utils.data.DataLoader"] = None, **kwargs):
        r""" Fits the encoder and decoder based on data. Note that a call to fit does not reset the weights in the
        networks that are currently in :attr:`encoder` and :attr:`decoder`.

        Parameters
        ----------
        data_loader : DataLoader
            Data loader which yields batches of data.
        n_epochs : int, default=5
            Number of epochs to train for.
        validation_loader : DataLoader, optional, default=None
            Data loader which yields batches of data for validation purposes. Can be
            left None, in which case no validation is performed.
        **kwargs
            Ignored kw.

        Returns
        -------
        self : Autoencoder
            Reference to self.
        """
        step = 0
        for epoch in range(n_epochs):

            self._encoder.train()
            self._decoder.train()
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch_0 = batch.to(device=self.device)

                self.optimizer.zero_grad()
                loss_value = self.evaluate_loss(batch_0)
                loss_value.backward()
                self.optimizer.step()

                self._train_losses.append((step, loss_value.item()))

                step += 1

            if validation_loader is not None:
                self._encoder.eval()
                self._decoder.eval()

                with torch.no_grad():
                    lval = []
                    for batch in validation_loader:
                        batch_0 = batch.to(device=self.device)

                        loss_value = self.evaluate_loss(batch_0).item()
                        lval.append(loss_value)
                    self._val_losses.append((step, np.mean(lval)))

        return self

    def fetch_model(self) -> AutoencoderModel:
        r""" Yields a new instance of :class:`AutoencoderModel`.

        .. warning::
            The model can be subject to side-effects in case :meth:`fit` is called multiple times, as no deep copy
            is performed of encoder and decoder networks.

        Returns
        -------
        model : AutoencoderModel
            The model.
        """
        from copy import deepcopy
        return AutoencoderModel(deepcopy(self._encoder), deepcopy(self._decoder), device=self.device, dtype=self.dtype)
