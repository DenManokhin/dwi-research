import numpy as np
from numfracpy import Mittag_Leffler_one


def adc_mlf_alpha(ADCs, alpha, b, delta):
    """ADC MLF alpha

    MLF_alpha(-ADCs * |k|^2 * t^alpha)
    k = sqrt(b / delta); t = delta
    """

    try:
        iter(b)
    except TypeError as te:
        b = [b]

    results = []
    for b_i in b:
        q = np.sqrt(b_i / delta)
        k, t = q, delta
        f = -ADCs * np.abs(k)**2 * t**alpha
        result = Mittag_Leffler_one(f, alpha)
        results.append(result)
    return np.asarray(results)