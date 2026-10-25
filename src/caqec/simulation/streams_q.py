"""Streams where BOTH the data-flip probability p_t and the measurement-flip probability q_t may vary."""
import numpy as np

from caqec.simulation.dataset import rows_from_realization
from caqec.simulation.nonstationary import GroundTruth, StreamBatch


def generate_streams_q(p_t, q_t, rounds: int, rng, n_streams: int = 1) -> StreamBatch:
    p_t, q_t = np.asarray(p_t, dtype=float), np.asarray(q_t, dtype=float)
    T, n = len(p_t), n_streams
    data = (rng.random((n, T, rounds, 3)) < p_t[None, :, None, None]).astype(np.uint8)
    meas = (rng.random((n, T, rounds, 2)) < q_t[None, :, None, None]).astype(np.uint8)
    real = np.concatenate([data.reshape(n * T, -1), meas.reshape(n * T, -1)], axis=1)
    X, _, y = rows_from_realization(real, rounds)
    return StreamBatch(observable=X.reshape(n, T, -1),
                       truth=GroundTruth(y=y.reshape(n, T), p_t=p_t, schedule=None,
                                         realization=real.reshape(n, T, -1)))