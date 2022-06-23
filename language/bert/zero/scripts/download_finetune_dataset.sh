#!/bin/bash

# Copyright (c) 2019-2020 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

git clone https://github.com/NVIDIA/DeepLearningExamples.git
mv DeepLearningExamples/PyTorch/LanguageModeling/BERT/data ./nv-dl-examples-data
rm -rf DeepLearningExamples
pip install wget

export BERT_PREP_WORKING_DIR=$PWD

python3 ./nv-dl-examples-data/bertPrep.py --action download --dataset squad
python3 ./nv-dl-examples-data/bertPrep.py --action download --dataset mrpc
python3 ./nv-dl-examples-data/bertPrep.py --action download --dataset sst-2