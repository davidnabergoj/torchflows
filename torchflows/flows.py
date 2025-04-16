import time
from copy import deepcopy
from typing import Union, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from torchflows.bijections.continuous.rnode import RNODE
from torchflows.bijections.base import Bijection
from torchflows.utils import flatten_event, unflatten_event, create_data_loader, get_batch_shape
from torchflows.base_distributions.gaussian import DiagonalGaussian


class BaseFlow(nn.Module):
    """Base normalizing flow class.
    """

    def __init__(self,
                 event_shape,
                 base_distribution: Union[torch.distributions.Distribution, str] = 'standard_normal'):
        """BaseFlow constructor.

        :param event_shape: shape of the event space.
        :param base_distribution: base distribution.
        """
        super().__init__()
        self.event_shape = event_shape
        self.event_size = int(torch.prod(torch.as_tensor(event_shape)))

        if base_distribution == 'standard_normal':
            self.base = DiagonalGaussian(loc=torch.zeros(self.event_size), scale=torch.ones(self.event_size))
        elif isinstance(base_distribution, torch.distributions.Distribution):
            self.base = base_distribution
        else:
            raise ValueError(f'Invalid base distribution: {base_distribution}')

        self.register_buffer('device_buffer', torch.empty(size=()))

    def get_device(self):
        """Returns the torch device for this object.
        """
        return self.device_buffer.device

    def base_log_prob(self, z: torch.Tensor):
        """Compute the log probability of input z under the base distribution.

        :param z: input tensor.
        :return: log probability of the input tensor.
        """
        zf = flatten_event(z, self.event_shape)
        log_prob = self.base.log_prob(zf)
        return log_prob

    def base_sample(self, sample_shape: Union[torch.Size, Tuple[int, ...]]):
        """Sample from the base distribution.

        :param sample_shape: desired shape of sampled tensor.
        :return: tensor with shape sample_shape.
        """
        z_flat = self.base.sample(sample_shape)
        z = unflatten_event(z_flat, self.event_shape)
        return z

    def regularization(self):
        """Compute the regularization term used in training.
        """
        return 0.0

    def fit_kl_p_to_q(self,
                      x_train: torch.Tensor,
                      x_val: torch.Tensor,
                      potential: callable,
                      n_epochs: int = 500,
                      lr: float = 0.05,
                      batch_size: int = 1024,
                      show_progress: bool = False,
                      keep_best_weights: bool = True,
                      early_stopping: bool = False,
                      early_stopping_threshold: int = 50,
                      time_limit_seconds: Union[float, int] = None):
        def loss_function(data, log_prob_target_data):
            return torch.mean(log_prob_target_data - self.log_prob(data))

        train_dataset = TensorDataset(x_train, -potential(x_train).detach())
        train_loader = DataLoader(train_dataset, batch_size=batch_size)

        val_dataset = TensorDataset(x_val, -potential(x_val).detach())
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        if len(list(self.parameters())) == 0:
            # If the flow has no trainable parameters, do nothing
            return

        self.train()
        t0 = time.time()
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        val_loss = None
        best_val_loss = torch.inf
        best_epoch = 0
        best_weights = deepcopy(self.state_dict())

        for epoch in (pbar := tqdm(range(n_epochs), desc='Fitting NF', disable=not show_progress)):
            if time_limit_seconds is not None and time.time() - t0 >= time_limit_seconds:
                print("Training time limit exceeded")
                break

            for train_batch in train_loader:
                optimizer.zero_grad()
                train_loss = loss_function(*train_batch)
                if not torch.isfinite(train_loss):
                    raise ValueError("Flow training diverged")
                train_loss += self.regularization()
                if not torch.isfinite(train_loss):
                    raise ValueError("Flow training diverged")
                train_loss.backward()
                optimizer.step()

                if show_progress:
                    if val_loss is None:
                        pbar.set_postfix_str(f'Training loss (batch): {train_loss:.4f}')
                    else:
                        pbar.set_postfix_str(
                            f'Training loss (batch): {train_loss:.4f}, '
                            f'Validation loss: {val_loss:.4f} [best: {best_val_loss:.4f} @ {best_epoch}]'
                        )

            # Compute validation loss at the end of each epoch
            # Validation loss will be displayed at the start of the next epoch
            if val_loader is not None:
                # Compute validation loss
                val_loss = 0.0
                for val_batch in val_loader:
                    val_loss += loss_function(*val_batch).detach()

                # Check if validation loss is the lowest so far
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch

                # Store current weights
                if keep_best_weights:
                    if best_epoch == epoch:
                        best_weights = deepcopy(self.state_dict())

                # Optionally stop training early
                if early_stopping:
                    if epoch - best_epoch > early_stopping_threshold:
                        break

        if val_loader is not None and keep_best_weights:
            self.load_state_dict(best_weights)

        # hacky error handling (Jacobian regularization is a non-leaf node within RNODE's autograd graph)
        if hasattr(self, 'bijection') and isinstance(self.bijection, RNODE):
            self.bijection.f.stored_reg = None

        self.eval()

    def fit(self,
            x_train: torch.Tensor,
            n_epochs: int = 500,
            lr: float = 0.05,
            batch_size: Union[int, str] = 1024,
            shuffle: bool = True,
            show_progress: bool = False,
            w_train: torch.Tensor = None,
            context_train: torch.Tensor = None,
            x_val: torch.Tensor = None,
            w_val: torch.Tensor = None,
            context_val: torch.Tensor = None,
            keep_best_weights: bool = True,
            early_stopping: bool = False,
            early_stopping_threshold: int = 50,
            max_batch_size_mb: int = None,
            time_limit_seconds: Union[float, int] = None):
        """Fit the normalizing flow to a dataset.

        Fitting the flow means finding the parameters of the bijection that maximize the probability of training data.
        Bijection parameters are iteratively updated for a specified number of epochs.
        If context data is provided, the normalizing flow learns the distribution of data conditional on context data.

        :param x_train: training data with shape `(n_training_data, *event_shape)`.
        :param n_epochs: perform fitting for this many steps.
        :param lr: learning rate. In general, lower learning rates are recommended for high-parametric bijections.
        :param batch_size: in each epoch, split training data into batches of this size and perform a parameter update for each batch.
        :param shuffle: shuffle training data. This helps avoid incorrect fitting if nearby training samples are similar.
        :param show_progress: show a progress bar with the current batch loss.
        :param w_train: training data weights with shape `(n_training_data,)`.
        :param context_train: training data context tensor with shape `(n_training_data, *context_shape)`.
        :param x_val: validation data with shape `(n_validation_data, *event_shape)`.
        :param w_val: validation data weights with shape `(n_validation_data,)`.
        :param context_val: validation data context tensor with shape `(n_validation_data, *context_shape)`.
        :param keep_best_weights: if True and validation data is provided, keep the bijection weights with the highest probability of validation data.
        :param early_stopping: if True and validation data is provided, stop the training procedure early once validation loss stops improving for a specified number of consecutive epochs.
        :param early_stopping_threshold: if early_stopping is True, fitting stops after no improvement in validation loss for this many epochs.
        :param int max_batch_size_mb: maximum batch size in megabytes.
        :param Union[float, int] time_limit_seconds: maximum allowed time for training.
        """
        t0 = time.time()
        self.train()

        if len(list(self.parameters())) == 0:
            # If the flow has no trainable parameters, do nothing
            return

        do_fit = False
        for p in self.parameters():
            if p.requires_grad:
                do_fit = True
                break
        if not do_fit:
            self.eval()
            return

        # Set the default batch size
        adaptive_batch_size = False
        if batch_size is None:
            batch_size = len(x_train)
        elif isinstance(batch_size, str) and batch_size == "adaptive":
            min_batch_size = max(32, min(1024, len(x_train) // 100))
            max_batch_size = min(4096, len(x_train) // 10)

            if max_batch_size_mb is not None:
                event_size_mb = self.event_size / 2 ** 20
                max_batch_size = max(1, min(max_batch_size, int(max_batch_size_mb / event_size_mb)))

            batch_size_adaptation_interval = 10  # double the batch size every 10 epochs
            adaptive_batch_size = True
            batch_size = min_batch_size

        # Process training data
        train_loader = create_data_loader(
            x_train,
            w_train,
            context_train,
            "training",
            batch_size=batch_size,
            shuffle=shuffle,
            event_shape=self.event_shape
        )

        # Process validation data
        if x_val is not None:
            val_loader = create_data_loader(
                x_val,
                w_val,
                context_val,
                "validation",
                batch_size=batch_size,
                shuffle=shuffle,
                event_shape=self.event_shape
            )

        def compute_batch_loss(batch_, reduction: callable = torch.mean):
            batch_x, batch_weights = batch_[:2]
            batch_context = batch_[2] if len(batch_) == 3 else None

            batch_log_prob = self.log_prob(batch_x.to(self.get_device()), context=batch_context)
            batch_weights = batch_weights.to(self.get_device())
            assert batch_log_prob.shape == batch_weights.shape, f"{batch_log_prob.shape = }, {batch_weights.shape = }"
            batch_loss = -reduction(batch_log_prob * batch_weights) / self.event_size

            return batch_loss

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        val_loss = None

        best_val_loss = torch.inf
        best_train_loss = torch.inf
        best_val_epoch = 0
        best_train_epoch = 0
        best_weights = deepcopy(self.state_dict())

        for epoch in (pbar := tqdm(range(n_epochs), desc='Fitting NF', disable=not show_progress)):
            if time_limit_seconds is not None and time.time() - t0 >= time_limit_seconds:
                print("Training time limit exceeded")
                break

            if (
                    adaptive_batch_size
                    and epoch % batch_size_adaptation_interval == batch_size_adaptation_interval - 1
                    and batch_size < max_batch_size
            ):
                batch_size *= 2
                batch_size = min(batch_size, max_batch_size)

                # Remake data loaders
                train_loader = create_data_loader(
                    x_train,
                    w_train,
                    context_train,
                    "training",
                    batch_size=batch_size,
                    shuffle=shuffle,
                    event_shape=self.event_shape
                )

                # Process validation data
                if x_val is not None:
                    val_loader = create_data_loader(
                        x_val,
                        w_val,
                        context_val,
                        "validation",
                        batch_size=batch_size,
                        shuffle=shuffle,
                        event_shape=self.event_shape
                    )

            _total_train_loss = 0
            _n_train_batches = 0
            for train_batch in train_loader:
                optimizer.zero_grad()
                train_loss = compute_batch_loss(train_batch, reduction=torch.mean)
                if not torch.isfinite(train_loss):
                    raise ValueError("Flow training diverged")
                _total_train_loss += float(train_loss)
                _n_train_batches += 1

                train_loss += self.regularization()
                if not torch.isfinite(train_loss):
                    raise ValueError("Flow training diverged")
                train_loss.backward()
                optimizer.step()

                if show_progress:
                    _train_string = f'Training loss (batch): {train_loss:.4f} [{best_train_loss:.4f} @ {best_train_epoch}]'
                    if val_loss is not None:
                        _val_string = f'Validation loss (batch): {val_loss:.4f} [{best_val_loss:.4f} @ {best_val_epoch}]'
                        _postfix_str = _train_string + ' , ' + _val_string
                    else:
                        _postfix_str = _train_string
                    pbar.set_postfix_str(_postfix_str)

            _average_train_loss = _total_train_loss / _n_train_batches
            if _average_train_loss < best_train_loss:
                best_train_loss = _average_train_loss
                best_train_epoch = epoch

            # Compute validation loss at the end of each epoch
            # Validation loss will be displayed at the start of the next epoch
            if x_val is not None:
                # Compute validation loss
                val_loss = 0.0
                for val_batch in val_loader:
                    val_loss += compute_batch_loss(val_batch, reduction=torch.sum).detach()
                val_loss /= len(x_val)

                # Check if validation loss is the lowest so far
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_epoch = epoch

            if keep_best_weights:
                if (
                        x_val is not None and best_val_epoch == epoch
                ) or (
                        x_val is None and best_train_epoch == epoch
                ):
                    best_weights = deepcopy(self.state_dict())

            # Optionally stop training early
            if early_stopping:
                if (
                        x_val is not None and (epoch - best_val_epoch) > early_stopping_threshold
                ) or (
                        x_val is None and (epoch - best_train_epoch) > early_stopping_threshold
                ):
                    break

        if keep_best_weights:
            self.load_state_dict(best_weights)

        # hacky error handling (Jacobian regularization is a non-leaf node within RNODE's autograd graph)
        if hasattr(self, 'bijection') and isinstance(self.bijection, RNODE):
            self.bijection.f.stored_reg = None

        self.eval()

    def variational_fit(self,
                        target_log_prob: callable,
                        n_epochs: int = 500,
                        lr: float = 0.05,
                        n_samples: int = 1,
                        early_stopping: bool = False,
                        early_stopping_threshold: int = 50,
                        keep_best_weights: bool = True,
                        show_progress: bool = False,
                        check_for_divergences: bool = False,
                        time_limit_seconds: Union[float, int] = None):
        """Train the normalizing flow to fit a target log probability.

        Stochastic variational inference lets us train a distribution using the unnormalized target log density instead of a fixed dataset.
        Refer to Rezende, Mohamed: "Variational Inference with Normalizing Flows" (2015) for more details (https://arxiv.org/abs/1505.05770, loss definition in Equation 15, training pseudocode for conditional flows in Algorithm 1).

        :param callable target_log_prob: function that computes the unnormalized target log density for a batch of
         points. Receives input batch with shape `(*batch_shape, *event_shape)` and outputs batch with
         shape `(*batch_shape)`.
        :param int n_epochs: number of training epochs.
        :param float lr: learning rate for the AdamW optimizer.
        :param float n_samples: number of samples to estimate the variational loss in each training step.
        :param bool show_progress: if True, show a progress bar during training.
        """
        t0 = time.time()

        if len(list(self.parameters())) == 0:
            # If the flow has no trainable parameters, do nothing
            return

        self.train()

        flow_training_diverged = False
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        best_loss = torch.inf
        best_epoch = 0
        initial_weights = deepcopy(self.state_dict())
        best_weights = deepcopy(self.state_dict())
        n_divergences = 0

        for epoch in (pbar := tqdm(range(n_epochs), desc='Fitting with SVI', disable=not show_progress)):
            if time_limit_seconds is not None and time.time() - t0 >= time_limit_seconds:
                print("Training time limit exceeded")
                break
            if check_for_divergences and not all([torch.isfinite(p).all() for p in self.parameters()]):
                flow_training_diverged = True
                print('Flow training diverged')
                print('Reverting to initial weights')
                break
            epoch_diverged = False
            optimizer.zero_grad()

            try:
                flow_x, flow_log_prob = self.sample(n_samples, return_log_prob=True)
                target_log_prob_value = target_log_prob(flow_x)
                loss = -torch.mean(target_log_prob_value + flow_log_prob)
                loss += self.regularization()

                if check_for_divergences:
                    if not torch.isfinite(loss):
                        epoch_diverged = True
                    if torch.max(torch.abs(flow_x)) > 1e8:
                        epoch_diverged = True
                    elif torch.max(torch.abs(flow_log_prob)) > 1e6:
                        epoch_diverged = True
                    elif torch.any(~torch.isfinite(flow_x)):
                        epoch_diverged = True
                    elif torch.any(~torch.isfinite(flow_log_prob)):
                        epoch_diverged = True

                if not epoch_diverged:
                    loss.backward()
                    optimizer.step()

                if not epoch_diverged:
                    with torch.no_grad():
                        if loss < best_loss:
                            best_loss = loss.detach()
                            best_epoch = epoch
                            if keep_best_weights:
                                best_weights = deepcopy(self.state_dict())
                        mean_flow_log_prob = flow_log_prob.mean().detach()
                        mean_target_log_prob = target_log_prob_value.mean().detach()
                else:
                    loss = torch.nan
                    mean_flow_log_prob = torch.nan
                    mean_target_log_prob = torch.nan
            except ValueError:
                epoch_diverged = True
                loss = torch.nan
                mean_flow_log_prob = torch.nan
                mean_target_log_prob = torch.nan

            n_divergences += epoch_diverged

            pbar.set_postfix_str(f'Loss: {loss:.4f} [best: {best_loss:.4f} @ {best_epoch}], '
                                 f'divergences: {n_divergences}, '
                                 f'flow log_prob: {mean_flow_log_prob:.2f}, '
                                 f'target log_prob: {mean_target_log_prob:.2f}')

            if epoch - best_epoch > early_stopping_threshold and early_stopping:
                break

        if flow_training_diverged:
            self.load_state_dict(initial_weights)
        elif keep_best_weights:
            self.load_state_dict(best_weights)

        # hacky error handling (Jacobian regularization is a non-leaf node within RNODE's autograd graph)
        if isinstance(self.bijection, RNODE):
            self.bijection.f.stored_reg = None

        self.eval()


class Flow(BaseFlow):
    """Normalizing flow class. Inherits from BaseFlow.

    This class represents a bijective transformation of a standard Gaussian distribution (the base distribution).
    A normalizing flow is itself a distribution which we can sample from or use it to compute the density of inputs.

    Implements: `sample`, `forward_with_log_prob`, `log_prob` (based on `forward_with_log_prob`)
    """

    def __init__(self, bijection: Bijection, **kwargs):
        """Flow constructor.

        :param Bijection bijection: transformation component of the normalizing flow.
        :param kwargs: keyword arguments passed to BaseFlow.
        """
        super().__init__(event_shape=bijection.event_shape, **kwargs)
        self.register_module('bijection', bijection)

    @property
    def context_shape(self):
        return self.bijection.context_shape

    def forward_with_log_prob(self, x: torch.Tensor, context: torch.Tensor = None):
        """Transform the input x to the space of the base distribution.

        :param torch.Tensor x: input tensor.
        :param torch.Tensor context: context tensor upon which the transformation is conditioned.
        :return: transformed tensor and the logarithm of the absolute value of the Jacobian determinant of the
         transformation.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        if context is not None:
            if self.context_shape is None:
                raise ValueError('Context shape must be set.')
            if self.event_shape is None:
                raise ValueError('Event shape must be set.')
            _batch_shape_1 = get_batch_shape(x, self.event_shape)
            _batch_shape_2 = get_batch_shape(context, self.context_shape)
            assert _batch_shape_1 == _batch_shape_2
            context = context.to(self.get_device())
        z, log_det = self.bijection.forward(x.to(self.get_device()), context=context)
        log_base = self.base_log_prob(z)
        return z, log_base + log_det

    def log_prob(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """Compute the logarithm of the probability density of input x according to the normalizing flow.

        :param torch.Tensor x: input tensor.
        :param torch.Tensor context: context tensor.
        :return: tensor of log probabilities.
        :rtype: torch.Tensor.
        """
        return self.forward_with_log_prob(x, context)[1]

    def sample(self,
               sample_shape: Union[int, torch.Size, Tuple[int, ...]],
               context: torch.Tensor = None,
               no_grad: bool = False,
               return_log_prob: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Sample from the normalizing flow.

        If context given, sample n tensors for each context tensor.
        Otherwise, sample n tensors.

        :param sample_shape: shape of tensors to sample.
        :param torch.Tensor context: context tensor with shape `c`.
        :param bool no_grad: if True, do not track gradients in the inverse pass.
        :param return_log_prob: if True, return log probabilities of sampled points as the second tuple component.
        :return: samples with shape `(*sample_shape, *event_shape)` if no context given or `(*sample_shape, *c, *event_shape)` if context given.
        :rtype: torch.Tensor
        """
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)

        if context is not None:
            z = self.base_sample(sample_shape=sample_shape)
            if get_batch_shape(context, self.context_shape) == sample_shape:
                # Option A: a context tensor is given for each sampled element
                pass
            else:
                # Option B: one context tensor is given for the entire to-be-sampled batch
                sample_shape = (*sample_shape, len(context))
                context = context[None].repeat(
                    *[*sample_shape, *([1] * len(context.shape))])  # Make context shape match z shape
                assert z.shape[:2] == context.shape[:2]
        else:
            z = self.base_sample(sample_shape=sample_shape)

        if no_grad:
            z = z.detach()
            with torch.no_grad():
                x, log_det = self.bijection.inverse(z.view(*sample_shape, *self.bijection.event_shape),
                                                    context=context)
        else:
            x, log_det = self.bijection.inverse(z.view(*sample_shape, *self.bijection.event_shape),
                                                context=context)
        x = x.to(self.get_device())

        if return_log_prob:
            log_prob = self.base_log_prob(z) + log_det
            return x, log_prob
        return x

    def regularization(self):
        if hasattr(self.bijection, 'regularization'):
            return self.bijection.regularization()
        else:
            return 0.0


class FlowMixture(BaseFlow):
    """Base class for mixtures of normalizing flows. Inherits from BaseFlow.

    A mixture uses flow objects as components, as well as their associated categorical distribution weights.
    It is a typical statistical mixture.
    """

    def __init__(self,
                 flows: List[Flow],
                 weights: List[float] = None,
                 trainable_weights: bool = False,
                 constrain_weights: bool = False):
        """FlowMixture constructor.

        :param List[Flow] flows: normalizing flow components.
        :param List[float] weights: mixture weights corresponding to flow components. All weights must be greater than 0. The sum of
            the weights must equal 1.
        :param bool trainable_weights: if True, makes the weights trainable.
        """
        super().__init__(event_shape=flows[0].event_shape)

        # Use uniform weights by default
        if weights is None:
            weights = [1.0 / len(flows)] * len(flows)

        self.constrain_weights = constrain_weights

        assert len(weights) == len(flows)
        assert all([w > 0.0 for w in weights])
        assert np.isclose(sum(weights), 1.0)

        self.flows = nn.ModuleList(flows)
        if trainable_weights:
            self.logit_weights = nn.Parameter(torch.log(torch.tensor(weights)))
        else:
            self.logit_weights = torch.log(torch.tensor(weights))

    @property
    def n_components(self):
        return len(self.flows)

    @property
    def weights(self):
        if self.constrain_weights:
            log_w_min = -5.0
            log_w_max = 5.0
            u = torch.sigmoid(self.logit_weights) * (log_w_max - log_w_min) + log_w_min
        else:
            u = self.logit_weights
        return torch.softmax(u, dim=0)

    @property
    def log_weights(self):
        return self.weights.log()

    def log_prob(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """Compute the log probability density of inputs x.

        :param torch.Tensor x: input tensor.
        :param torch.Tensor context: context tensor.
        :return: tensor of log probabilities.
        :rtype: torch.Tensor
        """
        flow_log_probs = torch.stack([flow.log_prob(x, context=context) for flow in self.flows])
        # (n_flows, *batch_shape)

        batch_shape = flow_log_probs.shape[1:]
        log_weights_reshaped = self.log_weights.view(-1, *([1] * len(batch_shape)))
        log_prob = torch.logsumexp(log_weights_reshaped + flow_log_probs, dim=0)  # batch_shape
        return log_prob

    def sample(self,
               sample_shape: int,
               context: torch.Tensor = None,
               no_grad: bool = False,
               return_log_prob: bool = False) -> torch.Tensor:
        """Sample from the flow mixture.

        :param int n: number of samples to draw.
        :param torch.Tensor context: context tensor.
        :param bool no_grad: if True, do not track gradients in the inverse pass during sampling.
        :param return_log_prob: if True, return log probabilities of sampled points as the second tuple component.
        :returns: tensor of drawn samples.
        :rtype: torch.Tensor
        """
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)

        flow_samples = []
        flow_log_probs = []
        for flow in self.flows:
            flow_x, flow_log_prob = flow.sample(sample_shape, context=context, no_grad=no_grad, return_log_prob=True)
            flow_samples.append(flow_x)
            flow_log_probs.append(flow_log_prob)

        flow_samples = torch.stack(flow_samples)  # (n_flows, n, *event_shape)
        categorical_samples = torch.distributions.Categorical(probs=self.weights).sample(
            sample_shape=sample_shape
        )  # (n,)
        one_hot = torch.nn.functional.one_hot(categorical_samples, num_classes=len(flow_samples)).T  # (n_flows, n)
        one_hot_reshaped = one_hot.view(*one_hot.shape, *([1] * len(self.event_shape)))
        # (n_flows, n, *event_shape)

        samples = torch.sum(one_hot_reshaped * flow_samples, dim=0)  # (n, *event_shape)

        if return_log_prob:
            flow_log_probs = torch.stack(flow_log_probs)  # (n_flows, n)
            log_weights_reshaped = self.log_weights[:, None]  # (n_flows, 1)
            log_prob = torch.logsumexp(log_weights_reshaped + flow_log_probs, dim=0)  # (n,)
            return samples, log_prob
        else:
            return samples

    def regularization(self):
        return sum([flow.regularization() for flow in self.flows])
