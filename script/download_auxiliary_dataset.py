#!/usr/bin/env python
"""
Usage:
    download_auxiliary_dataset.py DESTINATION_DIR

Options:
    -h --help   Show this screen.
"""

from collections import defaultdict
import glob
import os
from subprocess import call

from docopt import docopt
import pandas as pd


if __name__ == '__main__':
    args = docopt(__doc__)

    destination_dir = os.path.abspath(args['DESTINATION_DIR'])
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    os.chdir(destination_dir)
    call(['wget', 'http://www.phontron.com/download/conala-corpus-v1.1.zip', '-P', destination_dir, '-O', 'conala-corpus-v1.1.zip'])
    call(['unzip', '-o', 'conala-corpus-v1.1.zip'])
    call(['git', 'clone', '--depth=1', 'https://github.com/LittleYUYU/StackOverflow-Question-Code-Dataset.git'])
    call(['git', 'clone', '--depth=1', 'https://github.com/acmeism/RosettaCodeData.git'])

    files = glob.glob(os.path.join(destination_dir, 'RosettaCodeData/Lang/Java/**/*.java'))
    java_rosetta_code = defaultdict(list)
    for f in files:
        java_rosetta_code[f.split('/')[-2]].append(('/'.join(f.split('/')[-2:]), open(f, 'r').read()))

    pd.DataFrame.from_records([r[0] for r in java_rosetta_code.values()],
                              columns=['repo_path', 'content']).to_csv(os.path.join(destination_dir, 'java_rosetta_code.csv'), index=False)
