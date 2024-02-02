# Few-shot prediction of the experimental functional measurements for proteins with single point mutations

The code for quantitative prediction of single point mutations impact on experimentally measured protein function.

### Usage

First of all, data preprocessing is needed to run the model training.

After creation of pickled data with embeddings per protein (might take hours) the model can be run for training and evaluation.

### Required libraries

Our model data is created with ESM-1b embedder. 
It is used to extract embeddings from an protein sequences. Enables SOTA inference of structure. Released with [Rao et al. 2021](https://www.biorxiv.org/content/10.1101/2021.02.12.430858v2) 
(ICML'21 version, June 2021).


More info [here](https://github.com/facebookresearch/esm)

Pretrained LLM file (1.3Gb) for embedder is [here](https://dl.fbaipublicfiles.com/fair-esm/models/esm_msa1b_t12_100M_UR50S.pt), other file (contact regression) can be found in 'esm' folder.

### Data

MAVE data files are included in the 'mave' folder.

### Source code

The complete project is in Python plus PyTorch and it is in 'code' folder.

