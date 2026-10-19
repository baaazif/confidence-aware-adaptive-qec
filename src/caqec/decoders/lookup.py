# Syndrome (s1, s2) -> which data qubit to flip back (None = do nothing).
# This is the *most likely single-flip* explanation, not a guaranteed identification.
SYNDROME_TO_FLIP = {
    (0, 0): None,
    (1, 0): 0,
    (1, 1): 1,
    (0, 1): 2,
}


def parse_syndrome(bitstring: str) -> tuple:
    """Qiskit prints clbit 1 on the LEFT: '10' means s2=1, s1=0 -> (s1, s2) = (0, 1)."""
    return int(bitstring[1]), int(bitstring[0])


def compute_syndrome(bits) -> tuple:
    """Classical version of the circuit: (b0 XOR b1, b1 XOR b2)."""
    return bits[0] ^ bits[1], bits[1] ^ bits[2]


def decode_syndrome(syndrome: tuple):
    return SYNDROME_TO_FLIP[syndrome]


def apply_correction(bits, syndrome: tuple) -> tuple:
    flip = decode_syndrome(syndrome)
    out = list(bits)
    if flip is not None:
        out[flip] ^= 1
    return tuple(out)


def majority_vote(bits) -> int:
    return int(sum(bits) >= 2)