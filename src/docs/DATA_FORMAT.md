## Explanation of data format

Data is stored in [jsonlines](http://jsonlines.org/) format.  A prettified example of one row of this dataset is illustrated below.  Explanation of the keys are as follows:

- **repo:** the owner/repo
- **path:** the full path to the original file.
- **lineno:** the lineno in the original file the function or method came from.
- **func_name:** the function or method name
- **original_string:** the raw string before tokenization or parsing.
- **language:** the programming language.
- **code:** the part of the `original_string` that is code.
- **code_tokens:** tokenized version of `code`
- **docstring:** the top level comment or docstring, if exists in the original string.
- **docstring_tokens:** tokenized version of `docstring`.
- **sha:** the field is not being used
- **comment_tokens:** tokenized comments if they exist in the code.  This is not being used in the model at the moment. 
- **doc_id:** this is a unique id for tracking data lineage.  Not used in the model.
- **hash_key:** the value used to hash this datum (Only the part before the : is used).
- **hash_value:** the numeric hash value, used to split the data into train/valid/test sets.
- **partition:** a flag indicating what partition this datum belongs to {train, valid, test, etc.} This is not used by the model.  Instead we rely on directory structure to denote the partition of the data.

```{json}
{
  "repo": "github/myrepo",
  "path": "myrepo/some_code.py",
  "lineno": 167,
  "func_name": "country_list",
  "original_string": "def country_list(cts):\n    \"\"\"countries for comparisons\"\"\"\n    ct_nlp = []\n    for i in cts.keys():\n        nlped = nlp(i)\n        ct_nlp.append(nlped)\n    return ct_nlp\n",
  "language": "python",
  "code": "def country_list(cts):\n    \"\"\"\"\"\"\n    ct_nlp = []\n    for i in cts.keys():\n        nlped = nlp(i)\n        ct_nlp.append(nlped)\n    return ct_nlp\n",
  "code_tokens": [
    "def",
    "country_list",
    "cts",
    "\"\"\"\"\"\"",
    "ct_nlp",
    "for",
    "i",
    "in",
    "cts",
    "keys",
    "nlped",
    "nlp",
    "i",
    "ct_nlp",
    "append",
    "nlped",
    "return",
    "ct_nlp"
  ],
  "docstring": "countries for comparisons",
  "docstring_tokens": [
    "countries",
    "for",
    "comparisons"
  ],
  "sha": "",
  "comment_tokens": [],
  "doc_id": 162323,
  "hash_key": "github/myrepo:myrepo/some_code.py",
  "hash_val": 51889,
  "partition": "test"
}
```
