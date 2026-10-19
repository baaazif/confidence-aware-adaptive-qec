def unprotected_failure(p: float) -> float:
    """One qubit: logical failure = the qubit flipped."""
    return p


def protected_failure(p: float) -> float:
    """3-qubit code, perfect syndrome measurement: failure = 2 or 3 flips."""
    return 3 * p**2 * (1 - p) + p**3  # = 3p^2 - 2p^3


def any_physical_flip(p: float) -> float:
    """Probability that at least one of the 3 data qubits flipped."""
    return 1 - (1 - p) ** 3