[![Build Status](https://dev.azure.com/hahusain/hahusain/_apis/build/status/ml-msr-github.CodeSearchNet?branchName=master)](https://dev.azure.com/hahusain/hahusain/_build/latest?definitionId=4&branchName=master) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)  [![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

**Table of Contents**

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Introduction](#introduction)
  - [Project Overview](#project-overview)
  - [Data](#data)
  - [Network Architecture](#network-architecture)
  - [Evaluation](#evaluation)
    - [Annotations](#annotations)
  - [Benchmark](#benchmark)
  - [How to Contribute](#how-to-contribute)
- [Running the Baseline Model](#running-the-baseline-model)
  - [Setup](#setup)
  - [Training](#training)
  - [W&B Setup](#w&b-setup)
- [Data Details](#data-details)
  - [Data Acquisition](#data-acquisition)
  - [Preprocessed Data Format](#preprocessed-data-format)
  - [(Optional) Downloading Datasets from S3](#optional-downloading-datasets-from-s3)
    - [Preprocessed data](#preprocessed-data)
    - [All functions (w/o comments)](#all-functions--w-o-comments)
- [References](#references)
  - [Other READMEs](#other-readmes)
  - [License](#license)
  - [Important Documents](#important-documents)

<!-- /TOC -->

# Introduction

## Project Overview

  **CodeSearchNet** is a collection of datasets and a deep learning framework built on [TensorFlow](https://github.com/tensorflow/tensorflow) to explore the problem of code retrieval using natural language.  This research is a continuation of some ideas presented in this [blog post](https://githubengineering.com/towards-natural-language-semantic-code-search/) and is a joint collaboration between GitHub and the [Deep Program Understanding](https://www.microsoft.com/en-us/research/project/program/) group at [Microsoft Research - Cambridge](https://www.microsoft.com/en-us/research/lab/microsoft-research-cambridge/).  Our intent is to present and provide a platform for this research to the community by providing the following:

  1. Instructions for obtaining large corpora of relevant data
  2. Open source code for a range of baseline models, along with pre-trained weights
  3. Baseline evaluation metrics and utilities.
  4. Mechanisms to track progress on the community benchmark.  This is hosted by [Weights & Biases](https://www.wandb.com/), which is free for open source projects. TODO: link here to active benchmark?

We hope that CodeSearchNet is a good step towards engaging with the broader machine learning and NLP community towards developing new machine learning models that understand source code and natural language. Despite the fact that we described a specific task here, we expect and welcome many other uses of our dataset. 

More context regarding the motivation for this problem is in our blog post [TODO here](#TODO-TODO).

## Data

  The primary dataset consists of 2 million (`comment`, `code`) pairs from open source libraries.  Concretely, a `comment` is a top-level function or method comment (e.g. [docstrings](https://en.wikipedia.org/wiki/Docstring) in Python), and `code` is an entire function or method. Currently, the dataset contains Python, Javascript, Ruby, Go, Java, and PHP code.  Throughout this repo, we refer to the terms docstring and query interchangeably.  We partition the data into train, validation, and test splits such that code from the same repository can only exist in one partition. Currently this is the only dataset on which we train our model. Summary stastics about this dataset can be found in [this notebook](notebooks/ExploreData.ipynb)

## Network Architecture

  Our model ingests a parallel corpus of (`comments`, `code`) and learns to retrieve a code snippet given a natural language query.  Specifically, `comments` are top-level function and method comments (e.g. docstrings in Python), and `code` is an entire function or method. Throughout this repo, we refer to the terms docstring and query interchangeably.

  The query has a single encoder, whereas each programming language has its own encoder.  Our initial release has three languages: Python, Java, and C#. The available encoders are Neural-Bag-Of-Words, RNN, 1D-CNN, Self-Attention (BERT), and a 1D-CNN+Self-Attention Hybrid.

  The diagram below illustrates the general architecture of our model:
  
  ![alt text](images/architecture.png "Architecture")

## Evaluation

  The metric we use for evaluation is [Mean Reciprocal Rank](https://en.wikipedia.org/wiki/Mean_reciprocal_rank).  To calculate MRR, we use distractors from negative samples within a batch at evaluation time, with a fixed batch size of 1,000 (Note: we fix the batch size to 1,000 at evaluation time to standardize the MRR calculation, and do not do this at training time.)
    
  For example, consider a dataset of 10,005 (`comment`, `code`)  pairs.  For every (`comment`, `code`) pair in each of the 10 batches (we exclude the remaining 5 examples), we use the comment to retrieve the code, with the other code snippets in the batch serving as distractors.  We then average the MRR across all 10 batches to compute MRR for the dataset.  If the dataset is not divisible by 1,000, we exclude the final batch (any remainder that is less than 1,000) from the MRR calculation.
 
Since our model architecture is designed to learn a common representation for both code and natural language, we use the distances between these representations to rank results for the MRR calculation. We are computing distance using cosine similarity by default.

### Annotations
  We annotate retrieval results for the six languages from 99 general [queries](resources/queries.csv). This dataset will be used as groundtruth data for evaluation _only_. One task is to predict top 100 results per language per query from [all functions (w/o comments)](#all-functions--w-o-comments). [NDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) is computed as our main metrics.

## Benchmark

  We are using a community benchmark for this project to encourage collaboration and improve reproducibility.  It is hosted by [Weights & Biases](https://www.wandb.com/) (W&B), which is free for open source projects.  Our entries in the benchmark link to detailed logs of our training and evaluation metrics, as well as model artifacts, and we encourage other participants to provide as much transparency as possible. Here is the current state of the benchmark:

  We invite the community to improve on these baselines by submitting PRs with your new performance metrics.  TODO: how does the PR/submission flow interact between W&B leaderboard and this version? Please see these [instructions for submitting to the benchmark](src/docs/BENCHMARK.md).  Some requirements for submission:  

  1. Results must be reproducible with clear instructions.
  2. Code must be open sourced and clearly licensed.
  3. Model must demonstrate an improvement on at least one of the auxiliary tests.

## How to Contribute

  We are excited to offer the community useful tools&mdash;datasets, baseline models, and a collaboration forum via the Weights & Biases benchmark (TODO: link to active benchmark)&mdash;for the challenging research tasks of learning representations of code and code retrieval using natural language.  We encourage you to contribute by improving on our baseline models, sharing your ideas with others and [submitting your results to the collaborative benchmark](src/docs/BENCHMARK.md).
  
  We anticipate that the community will design custom architectures and use frameworks other than Tensorflow.  Furthermore, we anticipate that additional datasets will be useful.  It is not our intention to integrate these models, approaches, and datasets into this repository as a superset of all available ideas.  Rather, we intend to maintain the baseline models and a central place of reference with links to related repositories from the community.  TODO: link and description of W&B discussion forum?  We are accepting PRs that update the documentation, link to your project(s) with improved benchmarks, fix bugs, or make minor improvements to the code.  Here are [more specific guidelines for contributing to this repository](src/docs/CONTRIBUTING.md).  Please open an issue if you are unsure of the best course of action.  

# Running the Baseline Model

## Setup

  You should only have to perform the setup steps once to download the data and prepare the environment.

  1. Due to the complexity of installing all dependencies, we prepared Docker containers to run this code. You can find instructions on how to install Docker in the [official docs](https://docs.docker.com/get-started/).  Additionally, you must install [Nvidia-Docker](https://github.com/NVIDIA/nvidia-docker) to satisfy GPU-compute related dependencies.  For those who are new to Docker, this [blog post](https://towardsdatascience.com/how-docker-can-help-you-become-a-more-effective-data-scientist-7fc048ef91d5) provides a gentle introduction focused on data science.

  2. After installing Docker, you need to download the pre-processed datasets, which are hosted on S3.  You can do this by running `script/setup`.
      ```
      script/setup
      ```
      This will build Docker containers and download the datasets.  By default, the data is downloaded into the `resources/data/` folder inside this repository, with the directory structure described [here](resources/README.md).

  **The datasets you will download (most of them compressed) have a combined size of only ~ 3.5 GB.** 

  For more about the data, see [Data Details](#data-details) below as well as [this notebook](notebooks/ExploreData.ipynb).


## Training

This step assumes that you have a suitable Nvidia-GPU with [Cuda v9.0](https://developer.nvidia.com/cuda-90-download-archive) installed.  We used [AWS P3-V100](https://aws.amazon.com/ec2/instance-types/p3/) instances (a `p3.2xlarge` is sufficient).

  1. Start the model run environment by running `script/console`:
      ```
      script/console
      ```
      This will drop you into the shell of a Docker container with all necessary dependencies installed, including the code in this repository, along with data that you downloaded in the previous step.  By default you will be placed in the `src/` folder of this GitHub repository.  From here you can execute commands to run the model.

  2. Set up [W&B](https://docs.wandb.com/docs/started.html) (free for open source projects) per the instructions below if you would like to share your results on the community benchmark.  This is optional but highly recommended.

  3. The entry point to this model is `src/train.py`.  You can see various options by executing the following command:
      ```
      python train.py --help
      ```
      To test if everything is working on a small dataset, you can run the following command:
      ```
      python train.py --testrun
      ```

  4. Now you are prepared for a full training run.  Example commands to kick off training runs:
  * Training a neural-bag-of-words model on all languages
      ```
      python train.py --model neuralbow
      ```

    The above command will assume default values for the location(s) of the training data and a destination where would like to save the output model.  The default location for training data is specified in `/src/data_dirs_{train,valid,test}.txt`.  These files each contain a list of paths where data for the corresponding partition exists. If more than one path specified (separated by a newline), the data from all the paths will be concatenated together.  For example, this is the content of `src/data_dirs_train.txt`:

    ```
    $ cat data_dirs_train.txt
    ../resources/data/python/final/jsonl/train
    ../resources/data/javascript/final/jsonl/train
    ../resources/data/java/final/jsonl/train
    ../resources/data/php/final/jsonl/train
    ../resources/data/ruby/final/jsonl/train
    ../resources/data/go/final/jsonl/train
    ```

    By default models are saved in the `resources/saved_models` folder of this repository.

  * Training a 1D-CNN model on Python data only:
    ```
    python train.py --model 1dcnn /trained_models ../resources/data/python/final/jsonl/train ../resources/data/python/final/jsonl/valid ../resources/data/python/final/jsonl/test
    ```

    The above command overrides the default locations for saving the model to `trained_models` and also overrides the source of the train, validation, and test sets.

Additional notes:
* Options for `--model` are currently listed in `src/model_restore_helper.get_model_class_from_name`.

* Hyperparameters are specific to the respective model/encoder classes; a simple trick to discover them is to kick off a run without specifying hyperparameter choices, as that will print a list of all used hyperparameters with their default values (in JSON format).

* By default, models are saved in the `/resources/saved_models` folder of this repository, but this can be overridden as shown above.

## W&B Setup

 To initialize W&B:

   1. Navigate to the `/src` directory in this repository.

   2. If it's your first time using W&B on a machine, you will need to login:

      ```
      $ wandb login
      ```

   3. You will be asked for your api key, which appears on your [W&B profile page](https://app.wandb.ai/profile).

# Data Details

## Data Acquisition

 There are several options for acquiring the data.  

 1. Recommended: obtain the preprocessed dataset. 

   We recommend this option because parsing all the code from source can require a considerable amount of computation.  However, there may be an opportunity to parse, clean and transform the original data in new ways that can increase performance.  If you have run the setup steps above you will already have the preprocessed files, and nothing more needs to be done. The data will be available in the `/resources/data` folder of this repository, with [this directory structure](/resources/README.md).

 2. Extract the data from source and parse, annotate, and deduplicate the data.  To do this, see the [data extraction README](src/dataextraction/README.md).

## Preprocessed Data Format

TODO: consider moving this to a separate readme on data structure/format/directory structure?
Data is stored in [jsonlines](http://jsonlines.org/) format.  Each line in the uncompressed file represents one example (usually a function with an associated comment). A prettified example of one row is illustrated below.

- **repo:** the owner/repo
- **path:** the full path to the original file
- **func_name:** the function or method name
- **original_string:** the raw string before tokenization or parsing
- **language:** the programming language
- **code:** the part of the `original_string` that is code
- **code_tokens:** tokenized version of `code`
- **docstring:** the top level comment or docstring, if exists in the original string
- **docstring_tokens:** tokenized version of `docstring`
- **sha:** this field is not being used [TODO: add note on where this comes from?]
- **partition:** a flag indicating what partition this datum belongs to of {train, valid, test, etc.} This is not used by the model.  Instead we rely on directory structure to denote the partition of the data.
- **url:** the url for the this code snippet including the line numbers

Code, comments, and docstrings are extracted in a language-specific manner, removing artifacts of that language.

```{json}
{
  'code': 'def get_vid_from_url(url):\n'
          '        """Extracts video ID from URL.\n'
          '        """\n'
          "        return match1(url, r'youtu\\.be/([^?/]+)') or \\\n"
          "          match1(url, r'youtube\\.com/embed/([^/?]+)') or \\\n"
          "          match1(url, r'youtube\\.com/v/([^/?]+)') or \\\n"
          "          match1(url, r'youtube\\.com/watch/([^/?]+)') or \\\n"
          "          parse_query_param(url, 'v') or \\\n"
          "          parse_query_param(parse_query_param(url, 'u'), 'v')",
  'code_tokens': ['def',
                  'get_vid_from_url',
                  '(',
                  'url',
                  ')',
                  ':',
                  'return',
                  'match1',
                  '(',
                  'url',
                  ',',
                  "r'youtu\\.be/([^?/]+)'",
                  ')',
                  'or',
                  'match1',
                  '(',
                  'url',
                  ',',
                  "r'youtube\\.com/embed/([^/?]+)'",
                  ')',
                  'or',
                  'match1',
                  '(',
                  'url',
                  ',',
                  "r'youtube\\.com/v/([^/?]+)'",
                  ')',
                  'or',
                  'match1',
                  '(',
                  'url',
                  ',',
                  "r'youtube\\.com/watch/([^/?]+)'",
                  ')',
                  'or',
                  'parse_query_param',
                  '(',
                  'url',
                  ',',
                  "'v'",
                  ')',
                  'or',
                  'parse_query_param',
                  '(',
                  'parse_query_param',
                  '(',
                  'url',
                  ',',
                  "'u'",
                  ')',
                  ',',
                  "'v'",
                  ')'],
  'docstring': 'Extracts video ID from URL.',
  'docstring_tokens': ['Extracts', 'video', 'ID', 'from', 'URL', '.'],
  'func_name': 'YouTube.get_vid_from_url',
  'language': 'python',
  'original_string': 'def get_vid_from_url(url):\n'
                      '        """Extracts video ID from URL.\n'
                      '        """\n'
                      "        return match1(url, r'youtu\\.be/([^?/]+)') or \\\n"
                      "          match1(url, r'youtube\\.com/embed/([^/?]+)') or "
                      '\\\n'
                      "          match1(url, r'youtube\\.com/v/([^/?]+)') or \\\n"
                      "          match1(url, r'youtube\\.com/watch/([^/?]+)') or "
                      '\\\n'
                      "          parse_query_param(url, 'v') or \\\n"
                      "          parse_query_param(parse_query_param(url, 'u'), "
                      "'v')",
  'partition': 'test',
  'path': 'src/you_get/extractors/youtube.py',
  'repo': 'soimort/you-get',
  'sha': 'b746ac01c9f39de94cac2d56f665285b0523b974',
  'url': 'https://github.com/soimort/you-get/blob/b746ac01c9f39de94cac2d56f665285b0523b974/src/you_get/extractors/youtube.py#L135-L143'
}
```

Furthermore, summary statistics such as row counts and token length histograms can be found in [this notebook](notebooks/ExploreData.ipynb)

## (Optional) Downloading Data from S3

### Preprocessed data

The shell script `/script/setup` will automatically download these files into the `/resources/data` directory.  Here are the links to the relevant files for visibility:

The s3 links follow this pattern:

> https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{python,java,go,php,javascript,ruby}.zip

For example, the link for the `java` is:

> https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip

The size of the pre-processed dataset is 1.8 GB.

### All functions (w/o comments)

We also provide all functions (w/o comments), total ~6M functions. This data is located in the following S3 bucket:

> https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{python,java,go,php,javascript,ruby}_dedupe_definitions_v2.pkl

For example, the link for the python file is:

> https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python_dedupe_definitions_v2.pkl

The size of the raw filtered dataset is 17 GB.

# Related Projects

Here are related projects from the community that leverage these ideas.  PRs featuring other projects are welcome!

1. [Kubeflow](https://www.kubeflow.org/), [Code Search example](https://github.com/kubeflow/examples/tree/master/code_search).

2. Repository recommendations using [idi-o-matic](https://github.com/mkbehbehani/idi-o-matic).

# References

## Other READMEs

- [Submitting to the benchmark](src/docs/BENCHMARK.md)
- [Data extraction](src/dataextraction/README.md)
- [Data structure](/resources/README.md)

## License

This project is released under the [MIT License](LICENSE).

Container images built with this project include third party materials. See the [third party notice](src/docs/THIRD_PARTY_NOTICE.md) for details.

## Important Documents

- [Code of Conduct](src/docs/CODE_OF_CONDUCT.md)
- [Third Party Notice](src/docs/THIRD_PARTY_NOTICE.md)
