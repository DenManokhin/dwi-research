# dwi-research

Source code for paper:

D. Manokhin, Ya. Sokolovskyy. Application of Fractional Order Diffusion Model in Analysis of Diffusion-Weighted Magnetic Resonance Imaging Data // Computer Design Systems. Theory and Practice, vol. 7, no. 1, pp. 94-103, 2025, url: https://science.lpnu.ua/cds/all-volumes-and-issues/volume-7-number-1-2025/application-fractional-order-diffusion-model

## Usage

You can download samples from Connectome Diffusion Microstructure Dataset (CDMD) [here](https://springernature.figshare.com/search?q=MGH%20CDMD%20sub). Particularly sample sub_005 that was used in our research can be found [here](https://springernature.figshare.com/articles/dataset/MGH_CDMD_sub_005/16624702?file=30790042)

After you downloaded data, you can update paths in one of the following scripts and use them to fit respective model:

- `estimate_mono_adc.py` - for monoexponential model

- `estimate_mlf_alpha.py` - for MLF model

Other scripts can be used to analyze model fitting results and reproduce visualizations provided in the paper
