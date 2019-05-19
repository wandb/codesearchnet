#!/usr/bin/env python
"""
Usage:
    computerelevance.py [options] MODEL_PATH DATA_PATH TARGET_QUERIES OUTPUT_FILE

Standalone relevance evaluation script that outputs

Options:
    -h --help                        Show this screen.
    --distance-metric METRIC         The distance metric to use [default: cosine]
    --batch-size <VAL>               The batch size to use [default: 200]
    --quiet                          Less output (not one per line per minibatch). [default: False]
    --dryrun                         Do not log run into logging database. [default: False]
    --azure-info PATH                Azure authentication information file (JSON). Used to load data from Azure storage.
    --sequential                     Do not parallelise data-loading. Simplifies debugging. [default: False]
    --debug                          Enable debug routines. [default: False]
"""
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from docopt import docopt
from dpu_utils.utils import run_and_debug, RichPath
from more_itertools import chunked
from scipy.spatial.distance import cdist


import model_restore_helper
import model_test as test
from dataextraction.utils import tokenize_docstring_from_string


def run(arguments):
    azure_info_path = arguments.get('--azure-info', None)

    model = model_restore_helper.restore(path=arguments['MODEL_PATH'],
                                         is_train=False)

    # Encode all queries
    queries = []
    for query in RichPath.create(arguments['TARGET_QUERIES'], azure_info_path).read_as_jsonl():
        query['docstring_tokens'] = tokenize_docstring_from_string(query['query'])
        queries.append(query)

    queries = np.array(queries, dtype=np.object)
    batched_data = chunked(queries, int(arguments['--batch-size']))

    queries_per_lang = defaultdict(list)
    for batch_data in batched_data:
        query_representations = model.get_query_representations(batch_data)
        assert len(batch_data) == len(query_representations)
        for query, query_rep in zip(batch_data, query_representations):
            queries_per_lang[query['language']].append({
                'id': query['id'],
                'representation': query_rep
            })

    # Encode all code
    code_under_search = test.get_dataset_from([RichPath.create('DATA_PATH')])
    code_under_search = np.array(code_under_search, dtype=np.object)
    batched_data = chunked(code_under_search, int(arguments['--batch-size']))

    code_per_lang = defaultdict(list)
    for batch_data in batched_data:
        code_representations = model.get_code_representations(batch_data)
        assert len(batch_data) == len(code_representations)
        for code, code_rep in zip(batch_data, code_representations):
            code_per_lang[code['language']].append({
                'url': code['url'],
                'representation': code_rep
            })

    # Compute ranks and output

    def flatten_representations(list_of_dict: List[Dict[str, Any]]) -> np.ndarray:
        return np.stack(l['representation'] for l in list_of_dict)

    def evaluation():
        for language, queries in queries_per_lang.items():
            query_representations = flatten_representations(queries)

            candidate_snippets = code_per_lang[language]
            code_representations = flatten_representations(candidate_snippets)

            all_distances = cdist(query_representations, code_representations, metric=arguments['--distance-metric'])

            ranked_results = np.argsort(all_distances, axis=-1)

            for i in range(len(query_representations)):
                yield {'id': queries[i]['id'],
                        'ranked_results': [candidate_snippets[i][k] for k in ranked_results[i]]}

    target_output_file = RichPath.create(arguments['OUTPUT_FILE'], azure_info_path)
    target_output_file.save_as_compressed_file(evaluation())


if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args['--debug'])