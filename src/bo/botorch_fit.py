# botorch_fit.py

import time
from typing import Optional, Dict, Any, List, NamedTuple, Tuple
import torch
from torch.optim import Optimizer, Adam
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch import settings as gpt_settings

class OptimizationIteration(NamedTuple):
    iteration: int
    loss: float
    wall_time: float

def fit_gpytorch_torch(
    mll: MarginalLogLikelihood,
    optimizer_cls: Optimizer = Adam,
    options: Optional[Dict[str, Any]] = None,
    track_iterations: bool = True,
    approx_mll: bool = True,
) -> Tuple[MarginalLogLikelihood, List[OptimizationIteration]]:
    """
    Fits a GPyTorch model by maximizing the Marginal Log Likelihood (MLL) using a PyTorch optimizer.

    Args:
        mll (MarginalLogLikelihood): The MLL to be maximized.
        optimizer_cls (Optimizer): The PyTorch optimizer class to use (default is Adam).
        options (Optional[Dict[str, Any]]): Options for model fitting and the optimizer.
            Relevant options include:
                - "lr": Learning rate for the optimizer.
                - "maxiter": Maximum number of iterations.
                - "disp": If True, displays optimization progress.
        track_iterations (bool): If True, tracks the function values and wall time for each iteration.
        approx_mll (bool): If True, uses approximate MLL computation for efficiency.

    Returns:
        Tuple containing:
            - mll: The MLL with optimized parameters.
            - iterations: A list of OptimizationIteration objects with information on each iteration.
    """
    options = options or {}
    lr = options.get("lr", 0.1)
    maxiter = options.get("maxiter", 50)
    disp = options.get("disp", False)
    exclude = options.get("exclude", None)

    # Prepare parameters for optimization
    if exclude is not None:
        mll_params = [
            t for p_name, t in mll.named_parameters() if p_name not in exclude
        ]
    else:
        mll_params = list(mll.parameters())

    optimizer = optimizer_cls(
        params=[{"params": mll_params}],
        lr=lr,
    )

    iterations = []
    t_start = time.time()
    param_trajectory: Dict[str, List[torch.Tensor]] = {
        name: [] for name, param in mll.named_parameters()
    }
    loss_trajectory: List[float] = []

    mll.train()
    train_inputs = mll.model.train_inputs
    train_targets = mll.model.train_targets

    for i in range(maxiter):
        optimizer.zero_grad()
        with gpt_settings.fast_computations(log_prob=approx_mll):
            output = mll.model(*train_inputs)
            # Sum over batch dimensions for compatibility
            loss = -mll(output, train_targets).sum()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        loss_trajectory.append(loss_value)
        for name, param in mll.named_parameters():
            param_trajectory[name].append(param.detach().clone())

        if disp and ((i + 1) % 10 == 0 or i == maxiter - 1):
            print(f"Iter {i + 1}/{maxiter}: Loss = {loss_value:.4f}")

        if track_iterations:
            iterations.append(
                OptimizationIteration(
                    iteration=i,
                    loss=loss_value,
                    wall_time=time.time() - t_start,
                )
            )

    mll.eval()
    return mll, iterations
