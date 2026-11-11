"""Differentiable affine closed-loop response cost. No simulator or VLA claims."""
import torch


def response_chain(matrices, biases, initial_mean, initial_covariance,
                   disturbance_covariances, weights):
    """Return quadratic cost and moments after each physical transition.

    matrices: (T,d,d); biases: (T,d); disturbance_covariances: (T,d,d).
    weights[j] scores the state *after* transition j. Disturbances are assumed
    zero-mean, temporally independent, and independent of the initial state.
    Inputs may require gradients, including matrices (the feedback pathway).
    """
    horizon, dimension, last = matrices.shape
    if last != dimension or biases.shape != (horizon, dimension):
        raise ValueError("Invalid matrix/bias shapes")
    if disturbance_covariances.shape != matrices.shape or weights.shape != matrices.shape:
        raise ValueError("Invalid covariance/weight shapes")
    if initial_mean.shape != (dimension,) or initial_covariance.shape != (dimension, dimension):
        raise ValueError("Invalid initial moments")
    mean, covariance = initial_mean, initial_covariance
    means, covariances, costs = [], [], []
    for matrix, bias, disturbance, weight in zip(
            matrices, biases, disturbance_covariances, weights):
        mean = matrix @ mean + bias
        covariance = matrix @ covariance @ matrix.T + disturbance
        cost = mean @ weight @ mean + torch.trace(weight @ covariance)
        means.append(mean)
        covariances.append(covariance)
        costs.append(cost)
    return torch.stack(costs).sum(), torch.stack(means), torch.stack(covariances)
