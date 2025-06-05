import pyterrier as pt
import argparse
import json
import os.path
import sys

def parse_args():
    parser = argparse.ArgumentParser('index_pyterrier.py')

    parser.add_argument('-i', type=str, help='.jsonl to be used as input', required=True)
    parser.add_argument('-o', type=str, help='output directory', required=True)
    parser.add_argument('-f', help='index fields separately (title, text, keywords)', action='store_true')

    args = parser.parse_args()

    if not os.path.isfile(args.i):
        sys.exit(f"error: file/directory {args.i} not found")
    if not os.path.isdir(args.o):
        sys.exit(f"error: file/directory {args.o} not found")
    
    return args

def iter_jsonl(filename: str, transform):
    'Index an entity jsonl file, calling `transform` for every loaded json entity'

    with open(filename, 'rt') as file:
        for l in file:
            raw = json.loads(l)

            yield transform(raw)

def transform_raw(raw: dict[str, str]) -> dict[str,str]:
    'Transform an entity dictionary into another entity with a single text field'

    text = ' \n '.join([raw['title'], raw['text'], ' \n '.join(raw['keywords'])])
    return {'docno': raw['id'], 'text': text}

def transform_fields(raw: dict[str, str]) -> dict[str,str]:
    'Transform an entity dictionary into an entity with multiple fields'

    return {'docno': raw['id'],
            'title': raw['title'],
            'text': raw['text'], 
            'keywords': ' \n '.join(raw['keywords'])}

if __name__ == '__main__':
    args = parse_args()
    if not args.f: #Index raw data
        (pt.IterDictIndexer(args.o, meta={'docno': 20}, blocks=True, threads=8)
         .index(iter_jsonl(args.i, transform_raw)))
    else: #Also index fields
        (pt.IterDictIndexer(args.o, meta={'docno': 20}, 
                            text_attrs=['title', 'text', 'keywords'], 
                            fields=True, blocks=True, threads=8)
         .index(iter_jsonl(args.i, transform_fields)))