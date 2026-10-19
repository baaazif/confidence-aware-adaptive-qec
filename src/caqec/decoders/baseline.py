import numpy as np

from caqec.decoders.base import Decoder, check_features


class LookupDecoder(Decoder):
    """Conventional decoder: lookup/majority on the final perfect syndrome, history ignored.

    In the y-convention it never adds a logical flip, so it predicts y_hat = 0 for every row.
    Its failure rate on a dataset is therefore exactly the fraction of rows with y = 1.
    """
    name = "lookup"
    provides_confidence = False

    def predict_proba(self, X) -> np.ndarray:
        return np.zeros(len(check_features(X)))