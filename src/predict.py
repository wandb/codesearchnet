"""
Usage:
    predict.py [options] PREDICTION_FILE

> python predict.py --wandb-run github/codesearchnet/xxxxxxxx ../resources/model_predictions.csv
> python predict.py --model-file ../resources/saved_models/*.pkl.gz ../resources/model_predictions.csv

Options:
    -h --help                        Show this screen.
    --wandb-run PATH                 W&B run id.
    --model-file PATH                Local model path.

"""
import pickle
import re

from annoy import AnnoyIndex
from docopt import docopt
from dpu_utils.utils import RichPath
import pandas as pd
from tqdm import tqdm
import wandb

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

    run_id = None
    if args.get('--wandb-run', None):
        api = wandb.Api()
        run = api.run(args['--wandb-run'])
        model_file = [f for f in run.files() if f.name.endswith('gz')][0].download(replace=True)
        local_model_path = model_file.name
        run_id = args['--wandb-run'].split('/')[-1]
    elif args.get('--model-file', None):
        local_model_path = args['--model-file']
    else:
        raise Exception('Either --wandb-run or --model-file needs to be set')

    model_path = RichPath.create(local_model_path, None)
    model = model_restore_helper.restore(
        path=model_path,
        is_train=False,
        hyper_overrides={})

    predictions = []
    for language in ('python', 'go', 'javascript', 'java', 'php', 'ruby'):
        definitions = pickle.load(open('../resources/data/{}_dedupe_definitions_v2.pkl'.format(language), 'rb'))
        indexes = [{'code_tokens': d['function_tokens'], 'language': d['language']} for d in tqdm(definitions)]
        code_representations = model.get_code_representations(indexes)

        indices = AnnoyIndex(code_representations[0].shape[0])
        for index, vector in tqdm(enumerate(code_representations)):
            if vector is not None:
                indices.add_item(index, vector)
        indices.build(10)

        for query in queries:
            for idx, _ in zip(*query_model(query, model, indices, language)):
                predictions.append((query, language, definitions[idx]['identifier'], definitions[idx]['url']))

    df = pd.DataFrame(predictions, columns=['query', 'language', 'identifier', 'url'])
    df.to_csv(args['PREDICTION_FILE'], index=False)

    if run_id:
        wandb.init(id=run_id, resume=True)
        wandb.save(args['PREDICTION_FILE'])
