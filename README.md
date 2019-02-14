 # This is a staging repo to allow collaboration with OSS Partners.   
  ### The contents of the master branch will be moved to `github/codenet` without any commit history prior to launch.


 [![Build Status](https://dev.azure.com/hahusain/hahusain/_apis/build/status/ml-msr-github.semantic-search-research?branchName=master)](https://dev.azure.com/hahusain/hahusain/_build/latest?definitionId=1?branchName=master) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)  [![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

  # Overview

  **CodeNet** is a deep-learning based framework built on [TensorFlow](https://github.com/tensorflow/tensorflow) that we use to research the problem of code retrieval using natural language.  This research is a continuation of some ideas presented [here](https://githubengineering.com/towards-natural-language-semantic-code-search/) and is a joint collaboration between GitHub and the [Deep Program Understanding](https://www.microsoft.com/en-us/research/project/program/) group at [Microsoft Research - Cambridge](https://www.microsoft.com/en-us/research/lab/microsoft-research-cambridge/).


  The goals of this repository are to provide the community with the following:

  1. Instructions on how to obtain large corpora of data for research on this problem.
  2. A modeling framework that aids in reproducing our results.
  3. Baseline evaluation metrics, as well as utilities to aid in evaluation.
  4. Links to pre-trained models as well as detailed training run information.

More context regarding the motivation for this problem is in our blog post [TODO here](#TODO-TODO).

Our intention is not to maintain an open-source deep learning framework, but instead present an interesting machine learning problem and provide data and code so our results are reproduceable.

 ## General Architecture

 ![alt text](images/architecture.png "Architecture")

 - This model ingests a parallel corpus of (`comments`, `code`) and learns to retrieve a code snippet given a natural language query.  Specifically, `comments` are top-level function and method comments (ex: in python called docstrings), and `code` is either an entire function or method. Throughout this repo, we refer to the terms docstring and query interchangibly.
 - The query has a single encoder, whereas each programming language has its own encoder.
 - Encoders available are: Neural-Bag-Of-Words, RNN, 1D-CNN, Self-Attention (BERT), and a 1D-CNN+Self-Attention Hybrid.

 # Evaluation

 The metric we use for evaluation is [Mean Reciprocal Rank](https://en.wikipedia.org/wiki/Mean_reciprocal_rank).

 ## Primary Dataset
 Since we do not have a labeled dataset for semantic code search, we are using a proxy dataset that is a parallel corpus of (`comments`, `code`) to force code and natural language into the same vector space.  We group paritition the data into train/validation/holdout splits such that code from the same repository can only exist in one partition.

 ## Auxilary Tests
 In order to guide our progress we also evaluate our model on external datasets that more closely resemble semantic search, as well as other tasks that test our ability to learn representations of code.  Throughout the documentation, we refer to these as **Auxilary tests**.  An outline of these these tests are below:

#### Search Auxilary Tests
These tests use datasets that might more closely resemble natural language searches for code.

1. [CoNala](https://conala-corpus.github.io/): curated stack overflow data that is human-labeled with intent.  From this we construct a parallel corpus of (code, intent).

2. [StaQC](http://web.cse.ohio-state.edu/~sun.397/docs/StaQC-www18.pdf): another dataset manually curated from stack overflow with (code, question) pairs.


#### Other Auxilary Tests:

  1. **Function Name Prediction:**  we use our primary dataset and construct a task that attempts that attempts to retrieve the body of a function or method given the function or method name.
  2. [Rosetta Code](http://www.rosettacode.org/wiki/Rosetta_Code): We use data from this site to construct several parallel corpora that has pairs of code that accomplish the same task from the python, csharp, and java programming languages.  We use this parallel corpus to see if we can retrieve code in a different programming language that is the same to the one given.  For example, given a snippet of python code, we evaluate how well the representations learned by our model can retrieve code in java or csharp that accomplish the same task.

##  Leaderboard

The current leaderboard for this project can be seen below.  

**Authors**|**GitHub Repo**|**Notes**|**Primary Dataset MRR**|**FuncName MRR**|**CoNaLa MRR**|**StaQC MRR**|**Rosetta MRR**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
GitHub+Microsoft|[link](https://github.com/ml-msr-github/semantic-code-search)|Neural Bag of Words (cosine loss) |0.662|**0.419**|**0.259**|**0.168**|**0.123**
GitHub+Microsoft|[link](https://github.com/ml-msr-github/semantic-code-search)|1DCNN+SelfAttention|**0.757**|0.416|0.135|0.106|0.054

 We encourage the community to beat these baselines and submit a PR including your new benchmark in the above leaderboard. Please see our guidelines for contributing [here](Contributing.md).  Some requriements for submission to this leaderboard:

  - results must be reproduceable with clear instructions.
  - code must be open sourced and clearly licensed.
  - model must demonstrate an improvement on one of the 5 test metrics.

 You may notice that we have provided links in the **Notes** section of the leaderboard to a dashboard that shows detailed logging of our training and evaluation metrics, as well as  model artifacts for increased transperency.  We are utlizing [Weights & Biases](https://www.wandb.com/) (WandB), which is free for open-source projects.  While logging your models on this system is optional, we encourage participants who want to be included on this leaderboard to provide as much transperency as possible.  More instructions on how to enable **WandB** are below.


  ## Setup Notes

  1. Install dependencies as necessary from `src/requirements.txt` Alternatively, we also provide a [publicly hosted docker container](https://hub.docker.com/r/hamelsmu/ml-gpu-lite/).

  2. **Optional, If Using Azure:** This step is for people already using Azure, who want to store their artifacts there and is completely optional.  Create a file called `azure_auth.json` that contains the following information.
     **Warning**: do not check this file into GitHub!
     ```{json}
     {
         "semanticcodesearch": {
             "sas_token": "your-token-here",
             "cache_location": "/your/directory"
         }
     }
     ```
     This file is optionally used to interact with Azure storage. This uses the `RichPath` class from the`dpu-utils` package. If you find something not working or not well documented, open an issue/PR at the [GitHub project](https://github.com/microsoft/dpu-utils).

     The cache location is used to cache training/test data and can grow to ~3GB) and should be a location on a local disk. For reference, here are instructions on how to obtain a [SAS token](https://docs.microsoft.com/en-us/azure/storage/common/storage-dotnet-shared-access-signature-part-1)

  3. Setup [WandB](https://docs.wandb.com/docs/started.html) per the instructions below if you would like to share your results on their platform.  This is a recommended step as they are hosting the leaderboard for this task.

  ## Running The Code

  #### Get Data

 * To extract your own data, see the [dataextraction README](src/dataextraction/README.md).
 * The URLs of the standard datasets we have used for training can be found in `src/data_dirs_{train,valid_test}.txt`.

  #### Optional: Setup Wandb

 You need to initialize wandb:

   1. Navigate to the `semantic-code-search/src` directory.

   2. If its your first time on your remote machine using wandb you will need to login:
      ```
      $ wandb login
      ```

   3. You will be asked for your api key, which is shown on your [wandb profile page](https://app.wandb.ai/profile).

   4. Finally, initialize your wandb environment in `semantic-search-research/src`
      ```
      $ wandb init
 	    ```
      Make the following selections:
        - `Which team should we use?` -> **select `github`**
        - `Which project should we use?` -> **select `search`**

  #### Training The Models

  `src/train.py` is the entry point to train all models and do some preliminary testing.
  Example commands to kick off training runs:
  * Training a neural-bag-of-words model on all languages:
    ```
    python train.py --azure-info azure_auth.json --model neuralbow trained_models/ data_dirs_{train,valid,test}.txt
    ```
    **Note**: `data_dirs_{train,valid,test}.txt` are files containing a list of URLs with the current standard training data.

    This will store the trained model in `trained_models/${UNIQUE_ID}`. You can also store directly to Azure storage, simplifying the command to
    ```
    python train.py --azure-info azure_auth.json --model neuralbow
    ```

  * Training a 1D-CNN model on C# data only:
    ```
    python train.py --azure-info azure_auth.json --model 1dcnn trained_models/ azure://semanticcodesearch/csharpdata/Processed_Data/jsonl/{train,valid,test}
    ```

  `train.py --help` works and will give you an overview of available options.

  **Note:** Options for `--model` are currently listed in `src/model_restore_helper.get_model_class_from_name`.

  **Note:** Hyperparameters are specific to the respective model/encoder classes; a simple trick to discover them is to kick off a run without specifying hyperparameter choices, as that will print a list of all used hyperparameters with their default values (in JSON format).

## License

This project is released under the [MIT License](LICENSE).

Container images built with this project include third party materials. See [THIRD_PARTY_NOTICE.md](THIRD_PARTY_NOTICE.md) for details.
