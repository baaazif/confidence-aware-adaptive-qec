from math import sqrt


def wilson_interval(failures: int, shots: int, z: float = 1.96):
    """95% Wilson score interval for a binomial rate (z = 1.96).

    Unlike the plain 'rate +/- z*sqrt(rate(1-rate)/n)' formula, it stays inside
    [0, 1] and gives a sensible non-zero upper bound when failures = 0.
    """
    if shots <= 0:
        raise ValueError("shots must be positive")
    if not 0 <= failures <= shots:
        raise ValueError("failures must be between 0 and shots")
    phat = failures / shots
    denom = 1 + z**2 / shots
    centre = (phat + z**2 / (2 * shots)) / denom
    half = z * sqrt(phat * (1 - phat) / shots + z**2 / (4 * shots**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)