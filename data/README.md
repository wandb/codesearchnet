# This is an empty directory where you will download the training data, using the [/script/setup](/script/setup) script.

After downloading the data, the directory structure will look like this:

```
├── csharp
│   └── final
│       └── jsonl
│           ├── test
│           ├── train
│           └── valid
├── java
│   └── final
│       └── jsonl
│           ├── test
│           ├── train
│           └── valid
├── python
│   └── final
│       └── jsonl
│           ├── test
│           ├── train
│           └── valid
├── aux
│   ├── conala
│   ├── rosetta
│   └── staqc
└── saved_models
```

## Explanation of directory structure:

- `{csharp,java,python}\final\jsonl{test,train,valid}`:  these directories will contain multi-part [jsonl](http://jsonlines.org/) files with the data partitioned into train, valid, and test sets.  The tensorflow framework in this repository expects its data to be stored in this format, and will concatenate and shuffle these files appropriately.
- `aux\{conala,rosetta,staqc}`: these files correspond to the auxilary tests.  Please see the README at the root of the repository for more background on the auxilary tests.
- `saved_models`: this is the default destination where your models will be saved if you do not supply a destination.

## Data Format

See [this](docs/DATA_FORMAT.md) for documentation and an example of how the data is stored.
