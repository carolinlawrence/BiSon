# BiSon: Bidirectional Sequence Generation
This repository contains code for bidirectional sequence generation (BiSon).

Results have been published in (please cite if you use this repository):

Carolin Lawrence, Bhushan Kotnis, and Mathias Niepert. 2019.
[Attending to Future Tokens For Bidirectional Sequence Generation.](https://arxiv.org/abs/1908.05915)
In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 
Hong Kong, China.

## Content

| Section | Description |
|-|-|
| [Installation](#installation) | How to install the package |
| [Overview](#overview) | Overview of the package |
| [Implementing a new dataset](#dataset) |  | 
| [General Notes](#general) |  Additional useful information |

<a name="installation"></a>
## Installation
External libraries are: numpy, torch>=0.4.1, tqdm, boto3, requests, regex

This repository is compatible with the following three projects, 
which are required to reproduce the results reported in the paper:

- [Huggingface's BERT models](https://github.com/huggingface/pytorch-transformers):
    Should be compatible with any instance of the `pyotch-pretrained-bert` version, results from the paper have been produced using a fork of commit hash a5b3a89545bfce466dd977a9c6a7b15554b193b1
- [BLEU Script](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl):
    To get BLEU evaluation scores, the script needs to be downloaded and placed in
    `bison/evals/multi-bleu.perl` 
    and have execution rights (e.g. `chmod u+x bison/evals/multi-bleu.perl`)
- [Sharc Evaluation Script](https://sharc-data.github.io/):
    The official sharc evaluation script needs to be downloaded and placed in 
    `bison/evals/evaluator_sharc.py`. It can be found in the codalab of the Sharc dataset: https://worksheets.codalab.org/worksheets/0xcd87fe339fa2493aac9396a3a27bbae8/, search for "evaluator.py".

<a name="overview"></a>
## Overview
- Example files to call BiSon for either training or prediction for two datasets 
(ShARC and Daily Dialog) can be found in [`example_files`](./example_files).
Be sure to adjust the variable ```REPO_DIR``` to point to the path of your repository.

- BiSon specific implementations:
  - [`arguments.py`](./bison/arguments.py): Specifies all possible arguments for both BiSon:
    - [`GeneralArguments`](./bison/arguments.py#L24): General settings.
    - [`BisonArguments`](./bison/arguments.py#L152): BiSon specific settings.
  - [`bison_handler.py`](./bison/bison_handler.py): Calls all necessary functions for BiSon training and prediction.
  - [`masking.py`](./bison/masking.py): Handles the masking procedure. Get a masker by calling `get_masker` and passing [`BisonArguments`](./arguments.py#L152).
  Currently one masker is implemented:
    - [`GenerationMasking.py`](./bison/masking.py#L407): Places masks in Part B, where Part A is conditioning input 
    and Part B will be just placeholder tokens ([MASK])) at prediction time. 
    Masks can either be placed using a Bernoulli distribution (`--masking_strategy bernoulli`) with 
    a specified mean (`--distribution_mean`) 
    or using a Gaussian distribution (`--masking_strategy gaussian`) with
    a specified mean (`--distribution_mean`) and standard deviation (`--distribution_stdev`) 
  - [`model_helper.py`](./bison/model_helper.py): Sets up some general BiSon settings.
  - [`predict.py`](./bison/predict.py): Handles BiSon prediction.
  - [`train.py`](./bison/train.py): Handles BiSon training.
  - [`util.py`](./bison/util.py): Some utility function, e.g. for reading and writing files.

- Several implemented datasets. Get a data handler by calling `get_data_handler` 
  from [`datasets_factory.py`](./bison/dataset_handlers/datasets_factory.py) and passing [`BisonArguments`](./arguments.py#L152).
  
  The general class that all other datasets should inherit from:
  - [`datasets_bitext.py`](./bison/dataset_handlers/datasets_bitext.py): 
  Implements all necessary functions a data handler should have.
  It assumes a tab separate files as input, where everything prior to the tab becomes Part A and 
  everything after the tab becomes Part B. At prediction time, BiSon aims to predict Part B.

  Dialogue datasets:
  - [`datasets_sharc.py`](./bison/dataset_handlers/datasets_sharc.py): Implements the ShARC dataset.
  - [`datasets_daily.py`](./bison/dataset_handlers/datasets_daily.py): Implements the Daily Dialog dataset.

- main python file:
  - [`run_bison.py`](./run_bison.py): Main entry point for any BiSon training and prediction.

<a name="dataset"></a>
## Implementing a new dataset
To implement a new dataset, ensure that it inherits from [`BitextHandler`](./bison/dataset_handlers/datasets_bitext.py).
See the documentation of each function and determine if your dataset needs to overwrite this functionality or not.

<a name="general"></a>
## General Notes
  * The learning rate depends on the number of epochs, see warmup_linear function in `optimizer.py`:
    At the last update step the learning rate is 0.
    If a run finishes with its highest score in the last epoch, increasing the epoch counter 
    does not necessarily help because it completely modifies the learning rate.
  * Training cannot simply be restarted from a saved model because Adam's parameters are not saved.
  * When using the parameter --gradient_accumulation_steps, the value for batch_size should be the 
    truly desired batch size, e.g. we want a batch size of 16 but only 6 examples fit into GPU RAM, then:
    
    `--train_batch_size 16 --gradient_accumulation_steps 3`
