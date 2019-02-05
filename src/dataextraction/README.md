# Data Extraction

This document describes the format and process for extracting data from various languages.

## Data Format
Data is stored in gzipped [JSONL](http://jsonlines.org/) format.
Each line in the uncompressed file represents one example (usually a function) in the
following format:
```json
{
  "code": "a string with the original code segment",
  "code_tokens": ["List", "of", "code", "tokens", ...],
  "docstring": "the original string of code documentation (or other query) about the code",
  "docstring_tokens": ["List", "of", "docstring", "tokens", ...],
  "comment_tokens": ["List", "of", "tokens", "within", "comments", "but", "not", "the", "docstring", ...],
  "language": "programming language name",
  "repo": "user/project",
  "path": "the/path/to/the/file/in/the/repo",
  "lineno": 23,
  "func_name": "NameOfFunction",
  "sha": "Optional string containing the SHA of the repo when extracted"
}
```
The `repo` field usually refers to a GitHub repo, the `path` field is the file from
which the sample was extracted, and the `lineno` field is the first line in which the
example appears.
Code, comment and docstring are extracted in a language-specific manner, removing
artifacts of that language (_e.g._, XML comments in C#).
  
## Extracting Data

TODO: Give a concrete run-through here for at least one language (from raw dataset
 to cleaned / de-duplicated / split data)

To extract the data:
* Use GitHub's infrastructure, GCP to extract code from allowed repositories in .csv.
* Parse the code using the language-specific extractor (in `src/dataextraction`)
  and generate the raw `.jsonl.gz` files. Documentation for each language can
  be found in each folder.
* Run `dataextraction/dedup_split.py` to deduplicate and split the data into folds.


## Filtering Conventions
For all languages, we do some filtering of data to ensure that the samples we are
considering are meaningful. Concretely, we use the following rules:
* Methods with no documentation are removed..
* Functions that are less than 3 lines long are removed. This removes interface  
  declarations, short methods, getters/setters and unimplemented methods.
* Functions with less than 3 tokens in their documentation are removed. If easily
  recognizable, functions with auto-generated empty documentation should be
  removed as well.
* Test methods are removed. Heuristically, this is any method whose name includes
  the substring "test" (or "Test").
* We only keep the first segment of documentation. By segment we mean a sequence
  of tokens separated from the next segment by an empty line (_i.e._ matching
  `\n\s*\n`).