 [![Build Status](https://dev.azure.com/hahusain/hahusain/_apis/build/status/ml-msr-github.semantic-search-research?branchName=master)](https://dev.azure.com/hahusain/hahusain/_build/latest?definitionId=1?branchName=master) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)  [![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

  # Overview

  **CodeNet** is a deep-learning based framework built on [TensorFlow](https://github.com/tensorflow/tensorflow) that we use to research the problem of code retrieval using natural language.  This research is a continuation of some ideas presented [here](https://githubengineering.com/towards-natural-language-semantic-code-search/) and is a joint collaboration between GitHub and the [Deep Program Understanding](https://www.microsoft.com/en-us/research/project/program/) group at [Microsoft Research - Cambridge](https://www.microsoft.com/en-us/research/lab/microsoft-research-cambridge/).


  The goals of this repository are to provide the community with the following:

  1. Instructions on how to obtain large corpora of data for research on this problem.
  2. A modeling framework that aids in reproducing our results.
  3. Baseline evaluation metrics, as well as utilities to aid in evaluation.
  4. Links to pre-trained models as well as detailed training run information.

More context regarding the motivation for this problem is in our blog post [TODO here](#TODO-TODO).

Our intention is not to maintain an open-source deep learning framework, but instead present an interesting machine learning problem and provide data and code so our results are reproducible.

 ## General Architecture

 ![alt text](images/architecture.png "Architecture")
  
 - This model ingests a parallel corpus of (`comments`, `code`) and learns to retrieve a code snippet given a natural language query.  Specifically, `comments` are top-level function and method comments (ex: in python called docstrings), and `code` is either an entire function or method. Throughout this repo, we refer to the terms docstring and query interchangibly. 
 - The query has a single encoder, whereas each programming language has its own encoder (our initial release has three languages: Python, Java, and C#).
 - Encoders available are: Neural-Bag-Of-Words, RNN, 1D-CNN, Self-Attention (BERT), and a 1D-CNN+Self-Attention Hybrid.

 ## Evaluation

 The metric we use for evaluation is [Mean Reciprocal Rank](https://en.wikipedia.org/wiki/Mean_reciprocal_rank).

 ## Primary Dataset
 Since we do not have a labeled dataset for semantic code search, we are using a proxy dataset that is a parallel corpus of (`comments`, `code`) to force code and natural language into the same vector space.  We group paritition the data into train/validation/test splits such that code from the same repository can only exist in one partition.
 
 ## Auxilary Tests
 In order to guide our progress we also evaluate our model on external datasets that more closely resemble semantic search, as well as other tasks that test our ability to learn representations of code.  Throughout the documentation, we refer to these as **Auxilary tests**.  An outline of these these tests are below:

### Search Auxilary Tests
These tests use datasets that might more closely resemble natural language searches for code.

1. [CoNala](https://conala-corpus.github.io/): curated stack overflow data that is human-labeled with intent.  From this we construct a parallel corpus of (code, intent).

2. [StaQC](http://web.cse.ohio-state.edu/~sun.397/docs/StaQC-www18.pdf): another dataset manually curated from stack overflow with (code, question) pairs.


### Other Auxilary Tests:

  1. **Function Name Prediction:**  we use our primary dataset and construct a task that attempts to retrieve the body of a function or method given the function or method name.
  2. [Rosetta Code](http://www.rosettacode.org/wiki/Rosetta_Code): We use data from this site to construct several parallel corpora that has pairs of code that accomplish the same task from the python, csharp, and java programming languages.  We use this parallel corpus to see if we can retrieve code in a different programming language that is the same to the one given.  For example, given a snippet of python code, we evaluate how well the representations learned by our model can retrieve code in java or csharp that accomplish the same task.

##  Leaderboard

The current leaderboard for this project can be seen below.  

**Authors**|**GitHub Repo**|**Notes**|**Primary Dataset MRR**|**FuncName MRR**|**CoNaLa MRR**|**StaQC MRR**|**Rosetta MRR**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
GitHub+Microsoft|[link](https://github.com/ml-msr-github/semantic-code-search)|Neural Bag of Words (cosine loss) |0.662|**0.419**|**0.259**|**0.168**|**0.123**
GitHub+Microsoft|[link](https://github.com/ml-msr-github/semantic-code-search)|1DCNN+SelfAttention|**0.757**|0.416|0.135|0.106|0.054


 We encourage the community to beat these baselines and submit a PR including your new benchmark in the above leaderboard. Please see these [instructions for submitting to the leaderboard](src/docs/LEADERBOARD.md).  Some requirements for submission to this leaderboard:  

  - results must be reproducible with clear instructions.
  - code must be open sourced and clearly licensed.
  - model must demonstrate an improvement on at least one of the auxilary tests.

 You may notice that we have provided links in the **Notes** section of the leaderboard to a dashboard that shows detailed logging of our training and evaluation metrics, as well as  model artifacts for increased transperency.  We are utlizing [Weights & Biases](https://www.wandb.com/) (WandB), which is free for open-source projects.  While logging your models on this system is optional, we encourage participants who want to be included on this leaderboard to provide as much transperency as possible.  More instructions on how to enable **WandB** are below.


  ## Setup Notes
 
  1. Due to the complexity of installing all dependencies we prepared docker containers to run this code.  To get started, run the shell script `script/bootstrap`.  This will build a docker container that can be used to fetch the data.

  2. Run the shell script `script/setup`. This downloads the primary and auxilary datasets described above. The data is downloaded into the `data/` folder and will result in a directory structure described [here](data/README.md).
 
  3. Setup [WandB](https://docs.wandb.com/docs/started.html) per the instructions below if you would like to share your results on their platform.  This is a recommended step as they are hosting the leaderboard for this task.
 
  ## Running The Code	
 
  ### Get Data	
 
 There are several options for getting data.  
 
 1. Extract the data from source and parse, annotate, and dedup the data.  To do this, see the [dataextraction README](src/dataextraction/README.md).

 2. Obtain a pre-processed dataset.  (Recommended)

    Most people will want to use option 2 as parsing all of the code from source can require a considerable amount of computation.  However, there may be an opportunity to parse, clean and transform the original data in new ways that can increase performance.  If you have run the setup steps above you will already have the pre-processed files, and nothing more needs to be done.  You can read more about the format of the pre-processed data [here](src/docs/DATA_FORMAT.md).
 
  ### Optional: Setup Wandb	
 
 You need to initialize wandb:
 
   1. Navigate to the `/src` directory in this repository.
 
   2. If its your first time using wandb on a machine you will need to login:	

      ```
      $ wandb login
      ```

   3. You will be asked for your api key, which is shown on your [wandb profile page](https://app.wandb.ai/profile).
 
   4. Finally, initialize your wandb environment:	

      ```
      $ wandb init
 	    ```


  ### Training The Models	
 
  `/src/train.py` is the entry point to train all models and do some preliminary testing.
  Example commands to kick off training runs:
  * Training a neural-bag-of-words model on all languages:
    ```	
    python train.py --model neuralbow
    ```	

    The above command will assume default values for the location(s) of the training data and a destination where would like to save the model.  The default location for training data is specified in `/src/data_dirs_{train,valid,test}.txt`.  These files contain a list of paths where the data exists.  In the case that there is more than one path specified (seperated by a newline), then the data from all the paths will be concatenated together.  For example, this is the content of `src/data_dirs_train.txt`:

    ```
    $ cat data_dirs_train.txt
    ../data/python/final/jsonl/train
    ../data/csharp/final/jsonl/train
    ../data/java/final/jsonl/train
    ```
    
    By default models are saved in the `/data/saved_models` folder of this repository, however this can be overridden).
    

  * Training a 1D-CNN model on C# data only:
    ```
    python train.py --model 1dcnn trained_models/ ../data/csharp/final/jsonl/train ../data/csharp/final/jsonl/valid ../data/csharp/final/jsonl/test
    ```

    The above command overrides the default locations for saving the model to `trained_models` and also overrides the source of the train, validation, and test sets.

  `train.py --help` works and will give you an overview of available options.

  **Note:** Options for `--model` are currently listed in `src/model_restore_helper.get_model_class_from_name`.

  **Note:** Hyperparameters are specific to the respective model/encoder classes; a simple trick to discover them is to kick off a run without specifying hyperparameter choices, as that will print a list of all used hyperparameters with their default values (in JSON format).

### Saving Models

By default models are saved in the `/data/saved_models` folder of this repository, however this can be overridden.

## License

This project is released under the [MIT License](LICENSE).

Container images built with this project include third party materials. See [THIRD_PARTY_NOTICE.md](src/docs/THIRD_PARTY_NOTICE.md) for details.

## Important Documents:

- [Code of Conduct](src/docs/CODE_OF_CONDUCT.md)
- [Third Party Notice](src/docs/THIRD_PARTY_NOTICE.md)
- [Guidelines on Contributing](src/docs/CONTRIBUTING.md)
- [Instructions on Leaderboard Submissions](src/docs/LEADERBOARD.md)
- [Explanation of Data Format](src/docs/DATA_FORMAT.md)
