<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

## OPT and causal language modeling 

Fine-tuning (or training from scratch) the library models for language modeling on a text dataset for OPT... Models such as OPT are trained or fine-tuned using a causal language modeling
(CLM) loss.

The following example fine-tunes Facebook OPT on WikiText-2. We're using the raw WikiText-2 (no tokens were replaced before
the tokenization). The loss here is that of causal language modeling.

This training script is adapted from the [HuggingFace Language Modelling examples](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling).

You can invoke training by using the prepared bash script

```bash
bash ./run_clm.sh <batch-size-per-gpu> <mem-cap> <model> <gpu-num>
```

- batch-size-per-gpu: number of samples fed to each GPU, default is 16
- mem-cap: limit memory usage within a value in GB, default is 0 (no limit)
- model: the size of the OPT model, default is `6.7b`. Acceptable values include `125m`, `350m`, `1.3b`, `2.7b`, `6.7`, `13b`, `30b`, `66b`.
- gpu-num: the number of GPUs to use, default is 1.
