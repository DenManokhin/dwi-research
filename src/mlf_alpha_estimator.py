from functools import partial
import numpy as np
import scipy.stats as stats
from scipy.special import gamma

from src.CDMD_dataset import Sample
from src.models import adc_mlf_alpha
from src.fit import fit
from dwilib.fit import Model, Parameter
from dwilib.util import normalize_si_curve


def estimate_mlf_alpha(sample: Sample) -> dict:
    mlf_alpha_wrapper = partial(adc_mlf_alpha, delta=sample.delta)

    mlf_alpha_model = Model(
        'Alpha',
        'Normalized ADC alpha',
        mlf_alpha_wrapper,
        [
            Parameter('D_MLF', (0.5, 0.51, 1.0), (0, 1)),
            Parameter('Alpha', (0.5, 0.51, 1.0), (0.1, 1)),
        ],
        preproc=normalize_si_curve)
    results = fit(sample.image, sample.timepoints, mlf_alpha_model, sample.mask,
                  use_scipy=False, multiprocess=True)

    return results


def compute_entropy(sample: Sample, D: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    entropy = np.zeros_like(D)
    for i, j in np.argwhere(sample.mask != 0):
        p = adc_mlf_alpha(D[i, j], alpha[i, j], sample.timepoints, sample.delta)
        p_hat = p * np.conjugate(p)
        entropy[i, j] = stats.entropy(p_hat, base=len(p_hat), axis=0)
    return entropy


def compute_kurtosis(sample: Sample, alpha: np.ndarray) -> np.ndarray:
    K = 6 * (gamma(alpha + 1)**2 / gamma(2 * alpha + 1)) - 3
    K[sample.mask == 0] = 0
    return K