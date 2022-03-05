# Run T5 for language modelling With Colossal-AI

## Overview

In Colossal-AI, there are many ways to run T5 in a distributed manner. The `train_t5.py` script runs training with the specific configuration scripts in `t5_configs/` for different parallelisms of T5. We have provided some example configuration files of T5 and you can modify them to adapt to your own use.

## How to Prepare Webtext Dataset

We do not host any datasets for T5 training, however, we provide a detailed guide on how to prepare the dataset so that our results may be reproduced.

### Oveview
We utilize the publicly available [OpenWebText](https://github.com/eukaryote31/openwebtext) library by [jcpeterson](https://github.com/jcpeterson/openwebtext) and  [eukaryote31's](https://github.com/eukaryote31/openwebtext) work to download urls to different web pages. We then filtered, cleaned, and deduplicated all downloaded content according to the procedure described in Megatron's [openwebtext](./tools/openwebtext) directory. 

### Install necessary packages

**Note: LSH requires GCC's early version. We have tested that version 9.3.0 works, but version 10.3.0 is not.**

```bash
pip install ftfy langdetect numpy torch pandas nltk sentencepiece boto3 tqdm regex bs4 newspaper3k htmlmin tldextract cached-path
git clone https://github.com/mattilyra/LSH.git
cd LSH
python setup.py install
```

If you couldn't install it successfully, you may try to replace the `cMinhash.cpp` in `LSH/lsh` with ours, which is provided in `tools/lsh/cMinhash.cpp`.

### Download Data

1. Download the deduplicated URLs from [jcpeterson](https://mega.nz/#F!EZZD0YwJ!9_PlEQzdMVLaNdKv_ICNVQ!cc4RgQQZ).

1. Unzip the zip file and you will get a folder `URLs` which consists of many txt files including urls.

3. Remove blacklisted URLs. 

   *We appreciate Megatron-LM for making the data preprocessing code public. We have forked Megatron-LM and fixed some bugs. For your convenience, we have collated the needed files in `tools/Megatron`. Click [here](https://github.com/NVIDIA/Megatron-LM.git) to check the source code of Megatron-LM.*

   ```bash
   cd path/to/tools
   python Megatron/blacklist_urls.py <path/to/URLs> <path/to/clean_urls.txt>
   ```

4. Download the content from the clean urls and merge the contents into one loose json file with 1 json per newline of the format `{'text': text, 'url': unique_url}`. 

   *We have forked and modified [openwebtext](https://github.com/yet-another-account/openwebtext) as there are some bugs in it. For your convenience, we provide our modified version in `tools/download`.*
   
   ```bash
   python download/download.py <path/to/clean_urls.txt> --n_procs 50 --output <path/to/raw.json>
   ```

### Prepare Data for T5 Training

1. Perform ftfy, English detection and remove documents with less than 128 tokens. This step can be sharded and run on shards.

   ```bash
   python Megatron/cleanup_dataset.py <path/to/raw.json> <path/to/clean.json>
   ```
   
   Additional cleanup (e.g. remove documents less than 512 characters or dataset specific cleaning like stories, realnews datasets) can be done using `cleanup_fix_dataset.py`. More details can be found by running `python cleanup_fix_dataset.py --help`.
   
2. Using LSH, find possible duplicates and store them in a file for later processing. The code supports saving and loading fingerprints for recurrent deduplications, and is also multithreaded for faster processing. More details are can be found by `python find_duplicate.py --help`.

   ```bash
   python Megatron/find_duplicates.py --inputs <path/to/clean.json> url --output <path/to/process_stage_one.json>
   ```

3. Based on similarity measure defind inside function `is_similar` (default: 0.9), group urls that are similar. Basically, for each group, only one url we should keep and remove the rest.

   ```bash
   python Megatron/group_duplicate_url.py <path/to/process_stage_one.json> <path/to/process_stage_two.json>
   ```

4. Remove similar documents that were detected in the last step. The `dedup.json` is the data after deduplication.

   ```bash
   python Megatron/remove_group_duplicates.py <path/to/process_stage_two.json> <path/to/clean.json> <path/to/dedup.json>
   ```

5. shuffle the dataset.

   ```bash
   shuf <path/to/dedup.json> -o <path/to/train_data.json>
   ```

## **Usage**

```Bash
#!/usr/bin/env sh
export DATA=/path/to/train_data.json

torchrun --standalone --nproc_per_node=<num_gpus> train_t5.py --config=t5_configs/<config_file> --from_torch
```

You can copy it and save it as `run.sh`. Then use `bash ./run.sh` to run the script in your terminal.

Please modify `DATA`, `num_gpus` and `config_file` with the path to your dataset, the number of GPUs and the config file path, respectively.

## T5


Here are the T5 configs' default parameter:

| config     | scale | GPU* | batch  size | MiB of each GPU | TP  | PP  | DP  |
|------------| ----- | ---- | ----------- |-----------------| --- | --- | --- |
| t5-vanilla | small | 1    | 1           | ?               | 1   | 1   | 1   |

*\*Note: For GPUs, we use Nvidia A100 80G.*

**We set** `TENSOR_PARALLEL` `PIPELINE_PARALLEL` **and** `DATA_PARALLEL` **as small as it can be to run every demo with the least number of GPUs.**


### **Modify the config file**

#### **General**

There are some **general rules** when modifying the config files.

```Plain%20Text
TP denotes Tensor Parallel
PP denotes Pipeline Parallel
DP denotes Data Parallel

GPUS = TP * PP * DP
Where DP is autoseted
```

You can set the **batch size** and the **epoch** number by changing the number of 
`BATCH_SIZE` and  `NUM_EPOCHS`, respectively. Then, we will introduce the config file of each mode.

#### **Vanilla & Data Parallel**

`Vanilla` is the basic mode of T5 with no parallelism at all. However, if you use more than 1 GPU and TP * PP < no. of GPUs, Colossal-AI will **set DP for you** **automatically**.

#### **nvme**

If you want to use nvme, run `apt update && apt install libaio-dev` to prepare the environment and change the `nvme_path` in `zero = dict(...)`. Be aware of that `nvme_path` should be the path your local file system.