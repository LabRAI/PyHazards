"""Deep ensembles of any registered PyHazards model.

Lakshminarayanan, Pritzel & Blundell, "Simple and Scalable Predictive Uncertainty Estimation using
Deep Ensembles", NeurIPS 2017 (arXiv 1612.01474). The paper released no code; this module follows
its Section 2 and Algorithm 1:

- ``M`` members (default 5) of the same architecture, each from its own random initialisation
  (``seeds``; the builder seeds ``torch`` with ``seeds[m]`` before building member ``m``) and
  trained independently on the whole training set with its own data shuffling
  (:meth:`DeepEnsemble.fit`; the paper found bagging unnecessary);
- prediction is the uniformly weighted mixture of the members: classification averages the member
  probabilities, regression with mean/variance outputs moment-matches the Gaussian mixture,
  ``mu = mean_m mu_m`` and ``sigma^2 = mean_m (sigma_m^2 + mu_m^2) - mu^2``;
- optional adversarial training with the fast gradient sign method (:func:`fgsm_example`): each
  step minimises ``loss(x, y) + loss(x', y)`` with ``x' = x + epsilon * sign(grad_x loss)`` and
  ``epsilon`` 1% of the training-data range of each input dimension (:func:`input_range_epsilon`).

The uncertainty decomposition for classification (:func:`classification_uncertainties`) is the
``uncertainties`` function of the code of Kondylatos, Papadopoulos, Camps-Valls & Papoutsis,
"Uncertainty-Aware Deep Learning for Wildfire Danger Forecasting" (arXiv 2509.25017), repository
Orion-AI-Lab/uncertainty-wildfires, ``utils/train_functions.py`` (MIT License, Copyright (c) 2025
Orion Lab, commit fcfbfb894926d173e2f4cafc95b08f02cefe6e00). That paper's deep ensemble is 10 LSTMs
(``base_model="wildfire_forecasting", base_kwargs={"hidden_size": 128}``); tests/oracle checks the
aggregation against the reference with its 10 released members.

Outputs of :meth:`DeepEnsemble.forward` per ``task``:

- ``classification`` / ``segmentation`` with ``C >= 2`` channels along dim 1 (logits or
  log-probabilities; members are mapped to probabilities with a softmax over dim 1): the log of the
  mixture probability, same shape as one member's output (e.g. ``(batch, 2)`` for
  ``wildfire_forecasting``); ``exp`` gives the ensemble probabilities.
- ``classification`` / ``segmentation`` with one channel (binary logits, e.g.
  ``(batch, 1, H, W)``; members are mapped with a sigmoid): the logit of the mixture probability, so
  ``sigmoid`` of the output is the ensemble probability.
- ``regression`` / ``forecasting``: a tuple ``(mean, variance)``. With
  ``regression_output="gaussian"`` (the paper) every member outputs ``2 * K`` channels along dim 1,
  the ``K`` means followed by ``K`` raw variances that are mapped with
  ``softplus(.) + min_variance`` (the paper's footnote 2); build the base model with twice the
  output channels. With ``regression_output="point"`` members output point predictions and the
  variance is their spread across members (the MSE-ensemble heuristic the paper compares against).

:meth:`DeepEnsemble.predict_uncertainty` returns the ensemble prediction with its uncertainty terms.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSIFICATION_TASKS = ("classification", "segmentation")
REGRESSION_TASKS = ("regression", "forecasting")
REGRESSION_OUTPUTS = ("gaussian", "point")
DEFAULT_NUM_MEMBERS = 5  # Algorithm 1: "Recommended default values are M = 5"
MIN_VARIANCE = 1e-6  # footnote 2: softplus plus a minimum variance of 1e-6
ENTROPY_EPS = 1e-6  # ``e`` in uncertainty-wildfires' uncertainties()

Epsilon = Union[float, torch.Tensor]
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def classification_uncertainties(member_probs: torch.Tensor, eps: float = ENTROPY_EPS) -> Dict[str, torch.Tensor]:
    """Mixture prediction and uncertainty terms from member class probabilities.

    ``member_probs`` is ``(samples, batch, classes, ...)``: one probability map per ensemble member
    (and per MC-dropout pass). Returns, with the class dimension at dim 1:

    - ``probs``: the uniform mixture ``mean_m p_m``;
    - ``epistemic``: ``var_m p_m`` per class (population variance over the samples);
    - ``aleatoric``: ``mean_m p_m (1 - p_m)`` per class;
    - ``entropy``: ``-sum_c p log(p + eps)`` of the mixture (class dimension reduced);
    - ``mutual_information``: entropy minus the mean member entropy
      ``-mean_m sum_c p_m log(p_m + eps)``.

    These are the definitions of ``uncertainties`` in uncertainty-wildfires
    (``utils/train_functions.py``), with ``eps = 1e-6`` as there.
    """
    if member_probs.ndim < 3:
        raise ValueError(
            "classification_uncertainties expects member probabilities of shape (samples, batch, "
            f"classes, ...), got shape {tuple(member_probs.shape)}."
        )
    mean = member_probs.mean(dim=0)
    entropy = -torch.sum(mean * torch.log(mean + eps), dim=1)
    member_neg_entropy = torch.sum(member_probs * torch.log(member_probs + eps), dim=2).mean(dim=0)
    return {
        "probs": mean,
        "epistemic": member_probs.var(dim=0, unbiased=False),
        "aleatoric": (member_probs * (1 - member_probs)).mean(dim=0),
        "entropy": entropy,
        "mutual_information": entropy + member_neg_entropy,
    }


def gaussian_mixture_moments(means: torch.Tensor, variances: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean and variance of the uniform mixture of ``N(means[m], variances[m])`` over dim 0.

    ``sigma^2 = mean_m (sigma_m^2 + mu_m^2) - mu^2``, computed in the equivalent and numerically
    safer form ``mean_m sigma_m^2 + var_m mu_m``.
    """
    if means.shape != variances.shape:
        raise ValueError(f"means and variances must have the same shape, got {tuple(means.shape)} and {tuple(variances.shape)}.")
    return means.mean(dim=0), variances.mean(dim=0) + means.var(dim=0, unbiased=False)


def split_gaussian_output(output: torch.Tensor, min_variance: float = MIN_VARIANCE) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a mean/variance network output ``(batch, 2K, ...)`` into ``(mean, variance)``.

    The first ``K`` channels are the means, the last ``K`` raw variances mapped with
    ``softplus(.) + min_variance``.
    """
    if output.ndim < 2 or output.size(1) % 2:
        raise ValueError(
            "A mean/variance output needs an even number of channels along dim 1 (K means followed by "
            f"K raw variances), got shape {tuple(output.shape)}. Build the base model with twice the "
            "output channels, or use regression_output='point'."
        )
    mean, raw = output.chunk(2, dim=1)
    return mean, F.softplus(raw) + min_variance


def gaussian_nll_loss(output: torch.Tensor, target: torch.Tensor, min_variance: float = MIN_VARIANCE) -> torch.Tensor:
    """Equation 1 of the paper, averaged: ``log(sigma^2) / 2 + (y - mu)^2 / (2 sigma^2)``.

    ``output`` is a member's raw mean/variance output (see :func:`split_gaussian_output`) and
    ``target`` has the shape of the mean half.
    """
    mean, variance = split_gaussian_output(output, min_variance)
    if target.shape != mean.shape:
        raise ValueError(f"target shape {tuple(target.shape)} does not match the mean shape {tuple(mean.shape)}.")
    return (0.5 * torch.log(variance) + (target - mean) ** 2 / (2 * variance)).mean()


def input_range_epsilon(inputs: torch.Tensor, fraction: float = 0.01, dims: Sequence[int] = (0,)) -> torch.Tensor:
    """FGSM step size per input dimension: ``fraction`` of the training-data range (paper: 1%).

    The range ``max - min`` is taken over ``dims`` (default: the sample dimension), so every other
    position (feature, time step, pixel) gets its own epsilon. The reduced dimensions are kept with
    size 1, so the result broadcasts against a batch. NaNs are ignored; a dimension without finite
    values gets epsilon 0.
    """
    if fraction <= 0:
        raise ValueError(f"fraction must be positive, got {fraction}")
    if not inputs.is_floating_point():
        inputs = inputs.to(torch.get_default_dtype())
    dims = tuple(sorted({d % inputs.ndim for d in dims}))
    missing = torch.isnan(inputs)
    high = torch.where(missing, torch.full_like(inputs, -math.inf), inputs).amax(dim=dims, keepdim=True)
    low = torch.where(missing, torch.full_like(inputs, math.inf), inputs).amin(dim=dims, keepdim=True)
    return torch.nan_to_num(fraction * (high - low), nan=0.0, posinf=0.0, neginf=0.0)


def fgsm_example(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, loss_fn: LossFn, epsilon: Epsilon) -> torch.Tensor:
    """Fast gradient sign adversarial example ``x + epsilon * sign(grad_x loss(model(x), y))``.

    Only the input gradient is computed; parameter gradients are left untouched.
    """
    perturbed = inputs.detach().clone().requires_grad_(True)
    loss = loss_fn(model(perturbed), targets)
    (gradient,) = torch.autograd.grad(loss, perturbed)
    epsilon = torch.as_tensor(epsilon, dtype=inputs.dtype, device=inputs.device)
    return (perturbed + epsilon * gradient.sign()).detach()


@contextmanager
def _dropout_enabled(module: nn.Module) -> Iterator[None]:
    """Put the dropout layers in training mode (MC dropout), as ``enable_dropout`` in the reference."""
    dropouts = [m for m in module.modules() if "Dropout" in type(m).__name__]
    previous = [m.training for m in dropouts]
    for m in dropouts:
        m.train()
    try:
        yield
    finally:
        for m, mode in zip(dropouts, previous):
            m.train(mode)


class DeepEnsemble(nn.Module):
    """Uniform mixture of independently initialised and trained members (deep ensemble).

    Parameters
    ----------
    members:
        The ``M`` member networks (same architecture, distinct initialisations).
    task:
        ``classification`` / ``segmentation`` (probability mixture) or ``regression`` /
        ``forecasting`` (moment-matched Gaussian); see the module docstring for the outputs.
    seeds:
        The member seeds (used for initialisation by the builder and for data shuffling by
        :meth:`fit`); defaults to ``0 .. M-1``.
    regression_output:
        ``"gaussian"`` (members output means and raw variances) or ``"point"``.
    mc_dropout_passes:
        ``0`` (default) runs each member once. ``k > 0`` keeps the members' dropout layers active
        at prediction and pools ``k`` passes per member (``M * k`` samples), the "deep ensemble + MC
        dropout" of Chakravarty (2025) and the ``forward_passes`` / ``dropout`` options of the
        uncertainty-wildfires test script.
    min_variance:
        Added after the softplus of the variance outputs (``1e-6``).
    """

    def __init__(
        self,
        members: Sequence[nn.Module],
        task: str = "classification",
        seeds: Optional[Sequence[int]] = None,
        regression_output: str = "gaussian",
        mc_dropout_passes: int = 0,
        min_variance: float = MIN_VARIANCE,
    ):
        super().__init__()
        task = task.lower()
        if task not in CLASSIFICATION_TASKS + REGRESSION_TASKS:
            raise ValueError(f"DeepEnsemble supports task in {CLASSIFICATION_TASKS + REGRESSION_TASKS}, got {task!r}.")
        if len(members) < 1:
            raise ValueError("DeepEnsemble needs at least one member.")
        if regression_output not in REGRESSION_OUTPUTS:
            raise ValueError(f"regression_output must be one of {REGRESSION_OUTPUTS}, got {regression_output!r}.")
        if mc_dropout_passes < 0:
            raise ValueError(f"mc_dropout_passes must be >= 0, got {mc_dropout_passes}.")
        seeds = list(range(len(members))) if seeds is None else [int(seed) for seed in seeds]
        if len(seeds) != len(members) or len(set(seeds)) != len(seeds):
            raise ValueError(f"seeds must be {len(members)} distinct integers, got {seeds}.")
        self.members = nn.ModuleList(members)
        self.task = task
        self.seeds = seeds
        self.regression_output = regression_output
        self.mc_dropout_passes = int(mc_dropout_passes)
        self.min_variance = float(min_variance)

    @property
    def num_members(self) -> int:
        return len(self.members)

    @property
    def custom_fit_reason(self) -> str:
        """Why :class:`pyhazards.engine.Trainer` must not ``fit`` the ensemble as one network."""
        return (
            "deep_ensemble members are trained independently, each on its own loss (Algorithm 1 of "
            "Lakshminarayanan et al.); training the averaged output as one network is a different "
            "method. Use ensemble.fit(inputs, targets, loss_fn=..., optimizer_factory=...) or train "
            "each of ensemble.members with its own Trainer; Trainer.evaluate and Trainer.predict "
            "work on the ensemble."
        )

    def member_outputs(self, *inputs: Any, **kwargs: Any) -> List[torch.Tensor]:
        """Raw outputs of every member (and every MC-dropout pass), member by member.

        All arguments are passed to every member (e.g. U-TAE's ``batch_positions``).
        """
        outputs: List[torch.Tensor] = []
        for member in self.members:
            if self.mc_dropout_passes:
                with _dropout_enabled(member):
                    outputs.extend(member(*inputs, **kwargs) for _ in range(self.mc_dropout_passes))
            else:
                outputs.append(member(*inputs, **kwargs))
        for output in outputs:
            if not isinstance(output, torch.Tensor):
                raise ValueError(
                    f"deep_ensemble members must return one tensor, got {type(output).__name__}."
                )
        if any(output.shape != outputs[0].shape for output in outputs):
            raise ValueError(f"member outputs differ in shape: {[tuple(o.shape) for o in outputs]}.")
        if outputs[0].ndim < 2:
            raise ValueError(
                f"deep_ensemble expects member outputs with a channel dimension, (batch, C, ...), got shape {tuple(outputs[0].shape)}."
            )
        return outputs

    def _member_probs(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(outputs)
        if stacked.size(2) == 1:
            return torch.sigmoid(stacked)
        return torch.softmax(stacked, dim=2)

    def _regression_moments(self, outputs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        stacked = torch.stack(outputs)
        if self.regression_output == "point":
            mean, epistemic = stacked.mean(dim=0), stacked.var(dim=0, unbiased=False)
            return {"mean": mean, "variance": epistemic, "epistemic": epistemic, "member_means": stacked}
        means, variances = split_gaussian_output(stacked.flatten(0, 1), self.min_variance)
        means = means.unflatten(0, stacked.shape[:2])
        variances = variances.unflatten(0, stacked.shape[:2])
        mean, variance = gaussian_mixture_moments(means, variances)
        return {
            "mean": mean,
            "variance": variance,
            "epistemic": means.var(dim=0, unbiased=False),
            "aleatoric": variances.mean(dim=0),
            "member_means": means,
            "member_variances": variances,
        }

    def forward(self, *inputs: Any, **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        outputs = self.member_outputs(*inputs, **kwargs)
        if self.task in REGRESSION_TASKS:
            moments = self._regression_moments(outputs)
            return moments["mean"], moments["variance"]
        stacked = torch.stack(outputs)
        log_m = math.log(stacked.size(0))
        if stacked.size(2) == 1:
            log_p = torch.logsumexp(F.logsigmoid(stacked), dim=0) - log_m
            log_not_p = torch.logsumexp(F.logsigmoid(-stacked), dim=0) - log_m
            return log_p - log_not_p
        return torch.logsumexp(torch.log_softmax(stacked, dim=2), dim=0) - log_m

    def predict_uncertainty(self, *inputs: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """Ensemble prediction and uncertainty terms.

        Classification / segmentation: ``probs``, ``epistemic``, ``aleatoric``, ``entropy``,
        ``mutual_information`` (see :func:`classification_uncertainties`) and ``member_probs``
        ``(samples, batch, C, ...)``. For one-channel binary outputs the terms are those of the
        two-class distribution ``(1 - p, p)``, reported for the positive class.
        Regression: ``mean``, ``variance`` (mixture), ``epistemic`` (variance of the member means),
        ``member_means`` and, for Gaussian members, ``aleatoric`` (mean member variance) and
        ``member_variances``.
        """
        outputs = self.member_outputs(*inputs, **kwargs)
        if self.task in REGRESSION_TASKS:
            return self._regression_moments(outputs)
        member_probs = self._member_probs(outputs)
        if member_probs.size(2) == 1:
            terms = classification_uncertainties(torch.cat([1 - member_probs, member_probs], dim=2))
            for key in ("probs", "epistemic", "aleatoric"):
                terms[key] = terms[key][:, 1:2]
        else:
            terms = classification_uncertainties(member_probs)
        terms["member_probs"] = member_probs
        return terms

    def fit(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: LossFn,
        optimizer_factory: Callable[[Any], torch.optim.Optimizer],
        epochs: int = 1,
        batch_size: int = 100,
        adversarial_epsilon: Optional[Epsilon] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "DeepEnsemble":
        """Train every member independently (Algorithm 1) on all of ``inputs`` / ``targets``.

        Member ``m`` gets its own optimizer ``optimizer_factory(member.parameters())`` and its own
        data order, drawn from a generator seeded with ``seeds[m]``; minibatches of ``batch_size``
        (the paper used 100 with Adam). ``loss_fn(member_output, targets)`` must be a proper scoring
        rule for the member's output, e.g. ``nn.NLLLoss()`` for log-probabilities,
        ``nn.CrossEntropyLoss()`` for logits, :func:`gaussian_nll_loss` for mean/variance outputs.
        With ``adversarial_epsilon`` (a float or a tensor broadcasting against a batch of inputs, see
        :func:`input_range_epsilon`) each step minimises ``loss(x, y) + loss(x', y)`` with the FGSM
        example ``x'``. Returns ``self``; afterwards the members follow the ensemble's mode again
        (call ``ensemble.eval()`` before predicting, as for any module).
        """
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError(f"inputs and targets differ in length: shape {tuple(inputs.shape)} vs {tuple(targets.shape)}.")
        if epochs < 1 or batch_size < 1:
            raise ValueError(f"epochs and batch_size must be positive, got {epochs} and {batch_size}.")
        n_samples = inputs.shape[0]
        for member, seed in zip(self.members, self.seeds):
            member_device = torch.device(device) if device is not None else next(
                (p.device for p in member.parameters()), torch.device("cpu")
            )
            member.to(member_device).train()
            optimizer = optimizer_factory(member.parameters())
            generator = torch.Generator().manual_seed(seed)
            for _ in range(epochs):
                order = torch.randperm(n_samples, generator=generator)
                for start in range(0, n_samples, batch_size):
                    index = order[start : start + batch_size]
                    x_batch = inputs[index].to(member_device)
                    y_batch = targets[index].to(member_device)
                    optimizer.zero_grad()
                    loss = loss_fn(member(x_batch), y_batch)
                    if adversarial_epsilon is not None:
                        x_adv = fgsm_example(member, x_batch, y_batch, loss_fn, adversarial_epsilon)
                        loss = loss + loss_fn(member(x_adv), y_batch)
                    loss.backward()
                    optimizer.step()
            member.train(self.training)
        return self

    def extra_repr(self) -> str:
        return (
            f"task={self.task!r}, num_members={self.num_members}, seeds={self.seeds}, "
            f"regression_output={self.regression_output!r}, mc_dropout_passes={self.mc_dropout_passes}"
        )


def deep_ensemble_builder(
    task: str,
    base_model: str = "",
    base_kwargs: Optional[Dict[str, Any]] = None,
    num_members: int = DEFAULT_NUM_MEMBERS,
    seeds: Optional[Sequence[int]] = None,
    regression_output: str = "gaussian",
    mc_dropout_passes: int = 0,
    **kwargs: Any,
) -> DeepEnsemble:
    """Build a deep ensemble of ``num_members`` copies of the registered model ``base_model``.

    Member ``m`` is ``build_model(base_model, task=task, **base_kwargs)`` built right after
    ``torch.manual_seed(seeds[m])`` (default seeds ``0 .. num_members-1``), so it has the same
    initial weights as a single model built with that seed. The global RNG state is restored
    afterwards.
    """
    from .builder import build_model

    kwargs.pop("name", None)
    if kwargs:
        raise TypeError(
            f"deep_ensemble got unexpected arguments {sorted(kwargs)}; member arguments go in base_kwargs."
        )
    if not base_model:
        raise ValueError("deep_ensemble needs base_model, the registry name of the member model.")
    if base_model == "deep_ensemble":
        raise ValueError("deep_ensemble cannot use itself as base_model.")
    if num_members < 1:
        raise ValueError(f"num_members must be at least 1, got {num_members}.")
    seeds = list(range(num_members)) if seeds is None else [int(seed) for seed in seeds]
    if len(seeds) != num_members or len(set(seeds)) != num_members:
        raise ValueError(f"seeds must be {num_members} distinct integers, got {seeds}.")
    members = []
    with torch.random.fork_rng(devices=[]):
        for seed in seeds:
            torch.manual_seed(seed)
            members.append(build_model(base_model, task=task, **dict(base_kwargs or {})))
    return DeepEnsemble(
        members,
        task=task,
        seeds=seeds,
        regression_output=regression_output,
        mc_dropout_passes=mc_dropout_passes,
    )


__all__ = [
    "CLASSIFICATION_TASKS",
    "DEFAULT_NUM_MEMBERS",
    "DeepEnsemble",
    "REGRESSION_OUTPUTS",
    "REGRESSION_TASKS",
    "classification_uncertainties",
    "deep_ensemble_builder",
    "fgsm_example",
    "gaussian_mixture_moments",
    "gaussian_nll_loss",
    "input_range_epsilon",
    "split_gaussian_output",
]
