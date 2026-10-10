"""Pure affine-Beta moments without model or tensor dependencies."""


def beta_statistics(alpha: float, beta: float) -> tuple[float, float, float]:
    """Affine-Beta mean, concentration, and variance from one (alpha, beta) pair."""

    concentration = alpha + beta
    mean = 2.0 * alpha / concentration - 1.0
    variance = 4.0 * alpha * beta / (concentration * concentration * (concentration + 1.0))
    return mean, concentration, variance
