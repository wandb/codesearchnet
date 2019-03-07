# This is an empty directory where you will download the training data, using the [/script/setup](/script/setup) script.

After downloading the data, the directory structure will look like this:

```
├──data
|    ├── csharp
|    │   └── final
|    │       └── jsonl
|    │           ├── test
|    │           ├── train
|    │           └── valid
|    ├── java
|    │   └── final
|    │       └── jsonl
|    │           ├── test
|    │           ├── train
|    │           └── valid
|    ├── python
|    │   └── final
|    │       └── jsonl
|    │           ├── test
|    │           ├── train
|    │           └── valid
|    └── aux
|        ├── conala
|        ├── rosetta
|        └── staqc
└── saved_models
```

## Directory structure

- `{csharp,java,python}\final\jsonl{test,train,valid}`:  these directories will contain multi-part [jsonl](http://jsonlines.org/) files with the data partitioned into train, valid, and test sets.  The baseline training code uses TensorFlow, which expects data to be stored in this format, and will concatenate and shuffle these files appropriately.
- `aux\{conala,rosetta,staqc}`: datasets for the auxiliary tests.  Please see the README at the root of the repository for more background on the auxiliary tests.
- `saved_models`: default destination where your models will be saved if you do not supply a destination

## Data Format

See [this](docs/DATA_FORMAT.md) for documentation and an example of how the data is stored.
