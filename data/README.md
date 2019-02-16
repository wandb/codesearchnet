# This is an empty directory where you will download the training data, using the [/script/boostrap](/script/boostrap) script.

After downloading the data, the directory structure will look like this:

```
├── csharp
│   └── final
│       ├── DataFrame
│       └── jsonl
│           ├── holdout
│           ├── test
│           ├── train
│           └── valid
├── java
│   └── final
│       ├── DataFrame
│       └── jsonl
│           ├── holdout
│           ├── test
│           ├── train
│           └── valid
├── python
│   └── final
│       ├── DataFrame
│       └── jsonl
│           ├── holdout
│           ├── test
│           ├── train
│           └── valid
├── aux
│   ├── conala
│   ├── rosetta
│   │   └── Python
│   └── staqc
└── saved_models
```

## Explanation of directory structure:

- `{csharp,java,python}\final\jsonl{holdout,test,train,valid}`:  these directories will contain multi-part [jsonl](http://jsonlines.org/) files with the data partitioned into train, test, valid, and holdout sets.  The tensorflow framework in this repository expects its data to be stored in this format, and will concatenate and shuffle these files appropriately.
- `{csharp,java,python}\final\DataFrame`: contains pandas DataFrames with data collected into a tabular format to facilitate exploratory data analysis and summary statistics.  Each language has 4 dataframes corresponding to {holdout, test, train, valid} partitions. 
- `aux\{conala,rosetta,staqc}`: these files correspond to the auxilary tests.  Please see the README at the root of the repository for more background on the auxilary tests.
- `saved_models`: this is the default destination where your models will be saved if you do not supply a destination.

## Data Format

See [this](docs/DATA_FORMAT.md) for documentation and an example of how the data is stored.
