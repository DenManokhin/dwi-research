from src.CDMD_dataset import Sample
from src.fit import fit
from dwilib.models import Models


def estimate_mono_adc(sample: Sample) -> dict:

    mono_exp = [x for x in Models if x.name == "MonoN"][0]
    results = fit(sample.image, sample.timepoints, mono_exp, sample.mask,
                  use_scipy=False, multiprocess=True)

    return results
