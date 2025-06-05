import pyterrier as pt
import argparse
import json
import os.path
import sys

def parse_args():
    parser = argparse.ArgumentParser('index_pyterrier.py')

    parser.add_argument('-i', type=str, help='.jsonl to be used as input', required=True)
    parser.add_argument('-o', type=str, help='output directory', required=True)

    args = parser.parse_args()

    if not os.path.isfile(args.i):
        sys.exit(f"error: file/directory {args.i} not found")
    if not os.path.isdir(args.o):
        sys.exit(f"error: file/directory {args.o} not found")
    
    return args

def iter_jsonl(filename):
    with open(filename, 'rt') as file:
        for l in file:
            raw = json.loads(l)

            #Rewrite the raw document
            text = ' \n '.join([raw['title'], raw['text'], ' \n '.join(raw['keywords'])])
            yield {'docno': raw['id'], 'text': text}
    

if __name__ == '__main__':
    args = parse_args()
    index = pt.IterDictIndexer(args.o, meta={'docno': 20}, threads=8).index(iter_jsonl(args.i))