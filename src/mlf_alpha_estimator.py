from functools import partial
import numpy as np
from src.models import adc_mlf_alpha
from src.fit import fit
from dwilib.fit import Model, Parameter
from dwilib.util import normalize_si_curve


def estimate_mlf_alpha(data, mask, big_delta, small_delta):
    image = np.dstack([x["img"] for x in data])
    image = np.expand_dims(image, axis=2)
    timepoints = np.asarray([x["bval"] for x in data])
    delta = big_delta - small_delta / 3

    mlf_alpha_wrapper = partial(adc_mlf_alpha, delta=delta)

    mlf_alpha_model = Model(
        'Alpha',
        'Normalized ADC alpha',
        mlf_alpha_wrapper,
        [
            Parameter('ADCsN', (0.5, 0.51, 1.0), (0, 1)),
            Parameter('AlphaN', (0.5, 0.51, 1.0), (0.1, 1)),
        ],
        preproc=normalize_si_curve)
    results = fit(image, timepoints, mlf_alpha_model, mask,
                  use_scipy=False, multiprocess=True)

    return results
