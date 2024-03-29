# Few-shot prediction of the experimental functional measurements for proteins with single point mutations

The code for quantitative prediction of single point mutations impact on experimentally measured protein function.

### Usage

Data preprocessing is needed to run the model training.

Wild type and mutated sequence represetnations are generated using ESM-1b model [Rao et al. 2021](https://www.biorxiv.org/content/10.1101/2021.02.12.430858v2) 
(ICML'21 version, June 2021). <br/> [ESM embedder](https://github.com/facebookresearch/esm) files location must be specified for processing the sequence embeddings. 
<br/>The paths are defined in **utils.py**.

![image](https://github.com/bikman/fs-mutation-prediction/assets/82976389/543804b9-3b0a-4736-8e42-d79b083c202e)

The files can be downloaded from here: [Model ESM1b](https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt) and [Regression weights ESM1b](https://dl.fbaipublicfiles.com/fair-esm/regression/esm1b_t33_650M_UR50S-contact-regression.pt)

Run data creation script with argument specifying where to put the resulted dumps. 

Each dump file contains all the protein mutations, scores and embeddings.

```
run_prism_data_creation.py -dump_root [path_to_folder]
```

This will create pickled files:

![image](https://github.com/bikman/fs-mutation-prediction/assets/82976389/d1676825-3cc1-4730-a6fd-94017d8d7849)

To run a model for zero-shot pre-train process, specify where data dump files are located (parameter in **config.win.ini** or **config.linux.ini**)

![image](https://github.com/bikman/fs-mutation-prediction/assets/82976389/f46e4de4-7c59-4736-92e5-a3927977aa53)


After creation of pickled data with embeddings per protein (might take hours) the model can be run for training and evaluation.

Run model for zero-shot and fine-tuning, while **\[X\]** is an integer number from 1 to 39 according to a proteins list specified in **utils.py**

```
run_full_flow.py -ep [X] 
```

### Data

MAVE data files are included in the 'mave' folder.

### Source code

The complete project is in Python plus PyTorch and it is in 'code' folder.

### Configuration files

Several configuration files are provided for specifying various parameters of the system.

![image](https://github.com/bikman/fs-mutation-prediction/assets/82976389/b1720e55-acbb-4378-85c5-88797530ee8c)

One can spicify size of the model (number of channels, number of attention layers) and also training parameters like learning rate, number of epochs, patience, etc..

