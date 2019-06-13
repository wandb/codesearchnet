"""
Usage:
    predict.py LANGUAGE MODEL_FILE PREDICTION_FILE

> python predict.py python ../resources/saved_models/*.pkl.gz ../resources/python_predictions.csv

Options:
    -h --help                        Show this screen.

"""
import pickle
import re

from annoy import AnnoyIndex
from docopt import docopt
from dpu_utils.utils import RichPath
import pandas as pd
from tqdm import tqdm

from dataextraction.python.parse_python_data import tokenize_docstring_from_string
import model_restore_helper


def query_model(query, model, indices, language, topk=100):
    query_embedding = model.get_query_representations([{'docstring_tokens': tokenize_docstring_from_string(query),
                                                        'language': language}])[0]
    idxs, distances = indices.get_nns_by_vector(query_embedding, topk, include_distances=True)
    return idxs, distances


if __name__ == '__main__':
    args = docopt(__doc__)

    queries = pd.read_csv('../resources/queries.csv')
    queries = list(queries['query'].values)

    definitions = pickle.load(open('../resources/data/{}_dedupe_definitions_v2.pkl'.format(args['LANGUAGE']), 'rb'))

    model_path = RichPath.create(args['MODEL_FILE'], None)
    model = model_restore_helper.restore(
        path=model_path,
        is_train=False,
        hyper_overrides={})
    model_name = re.search(".*'(.*)'.*", str(type(model))).group(1).split('.')[-1]

    indexes = [{'code_tokens': d['function_tokens'], 'language': d['language']} for d in tqdm(definitions)]
    code_representations = model.get_code_representations(indexes)

    indices = AnnoyIndex(code_representations[0].shape[0])
    for index, vector in tqdm(enumerate(code_representations)):
        if vector is not None:
            indices.add_item(index, vector)
    indices.build(10)

    predictions = []
    for query in queries:
        for idx, _ in zip(*query_model(query, model, indices, args['LANGUAGE'])):
            predictions.append((query, model_name, args['LANGUAGE'], definitions[idx]['identifier'],
                                definitions[idx]['function'], definitions[idx]['docstring_summary'], definitions[idx]['url']))

    df = pd.DataFrame(predictions, columns=['query', 'model_name', 'language', 'identifier', 'function', 'docstring_summary', 'url'])
    df.to_csv(args['PREDICTION_FILE'], index=False)
