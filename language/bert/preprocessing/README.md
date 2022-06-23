## BERT Data Preprocessing

This folder contains tools used for BERT Pretraining and Fine-tuning. 
This section is adapted from the famous [DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT) repository and we appreciate their efforts for giving such detailed instructions. 
This section is made more modular and more friendly for the users to follow. 
We have also refactored the BERT dataset classes in Titans for compatibility with Colossal-AI.


## Requirements

Python 3.8 and above is needed for this BERT example. You can use the following command to create a new conda environment.

```shell
conda create -n bert python=3.8
conda activate bert
```

## Setup

Before we delve into the preprocessing procedure, let's prepare the environment for data processing.

1. Install jemalloc

Jemalloc is a high-perf memory allocator to speed up the process.

```shell
conda install -y jemalloc
```

2. Install Nvidia tools.

```shell
pip install git+https://github.com/NVIDIA/DeepLearningExamples.git#subdirectory=Tools/lddl
pip install git+https://github.com/NVIDIA/dllogger#egg=dllogger
```

3. Install other dependencies.

```shell
conda install -y -c conda-forge mpi4py openmpi
pip install tqdm boto3 requests six ipdb h5py nltk progressbar tokenizers
```

4. download punkt for nltk

```shell
python -m nltk.downloader punkt
```

A collated script can be as follows:

```shell
conda install -y jemalloc
pip install git+https://github.com/NVIDIA/DeepLearningExamples.git#subdirectory=Tools/lddl
pip install git+https://github.com/NVIDIA/dllogger#egg=dllogger
conda install -y -c conda-forge mpi4py openmpi
pip install tqdm boto3 requests six ipdb h5py nltk progressbar tokenizers
python -m nltk.downloader punkt
```

## Pretraining

We align with the BERT example in [DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT) and conduct pretraining in 2 phases.

- Phase 1: train on sequence length 128
- Phase 2: train on sequence length 512

### Directory Structure

```text
-- preprocessing
    -- pretrain
        -- phase1
        -- phase2
    -- wikipedia
        -- extracted
        -- source
        -- wikicorpus-en.xml.bz2
    pretrain_preprocess.sh
    README.md
```

### Procedure

1. Prepare Directory

Create the `pretrain` and `wikipedia` folders in the current directory. If you want to put the dataset and processed outputs in your data-storage directory, you can create a symbolic link instead. This is for users who have code in `HOME` but usually store large data in a specific directory such as `/data`.

2. Download and extract Wikipedia datasets

This will download the Wikipedia dataset into the `wikipedia` folder and extract the text from the file. 
The texts are then processed so that they can be fed to dask.
This could take hours.

```shell
download_wikipedia --outdir $PWD/wikipedia
```

3. Preprocess the data with dask

Use the following commands to preprocess the data with Dask. The first argument is the phase number, the second argument is the vocab file name and the third argument is the sequence length.

```shell
# phase 1
bash ./pretrain_preprocess.sh 1 bert-large-uncased 128

# phase 2
bash ./pretrain_preprocess.sh 2 bert-large-uncased 512
```

> **Note**  
> 1. Vocab file can be either a vocab file name or path to a vocab file.
> 2. You may see some dask errors during preprocessing, but it is ok as long as the program does not stop.
> 3. You can increase `num_dask_workers` to speed up data processing, but will consume more memory.
