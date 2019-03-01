#!/usr/bin/env python
"""
Usage:
    download_dataset.py DESTINATION_DIR

Options:
    -h --help   Show this screen.
"""

import os
from subprocess import call

from docopt import docopt


if __name__ == '__main__':
    args = docopt(__doc__)

    destination_dir = os.path.abspath(args['DESTINATION_DIR'])
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for language in ('python', 'csharp', 'java'):
        for split in ('train', 'valid', 'test'):
            language_dir = os.path.join(destination_dir, language, 'final', 'jsonl')
            if not os.path.exists(language_dir):
                os.makedirs(language_dir)
            os.chdir(language_dir)
            call(['wget', 'https://s3.amazonaws.com/code-search-net/CodeSearchNet/{}/{}.zip'.format(language, split), '-P', language_dir, '-O', '{}.zip'.format(split)])
            call(['unzip', '{}.zip'.format(split)])
            call(['rm', '{}.zip'.format(split)])
