import argparse
import os.path
import sys

import pyterrier as pt
from pyterrier_anserini import AnseriniIndex
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(prog='searcher.py')

    parser.add_argument('-q', type=str, help='.csv input file, containing queries', required=True)
    parser.add_argument('-i', type=str, help='Index containing the Anserini index', required=True)
    parser.add_argument('-o', type=str, help='Output file location', required=True)

    args = parser.parse_args()

    if not os.path.isfile(args.q):
        sys.exit(f'error: file/directory not found: {args.q}')
    if not os.path.isdir(args.i):
        sys.exit(f'error: directory not found: {args.i}')
    if not os.path.isdir(args.o):
        sys.exit(f'error: directory not found {args.o}')

    return args

if __name__ == '__main__':
    args = parse_args()

    #Load index
    index = AnseriniIndex(args.i)
    bm25 = index.retriever("BM25", num_results=100)

    #Load queries
    queries = pd.read_csv(args.q)

    #Transform applies the search
    df_ans = bm25.transform(queries)

    #Detailed result
    df_ans.to_csv(f'{args.o}/raw_results.csv', index=False)

    #Formatted result, for submitting to kaggle
    ( df_ans[['qid', 'docno']]
        .rename(columns={'qid': 'QueryId', 'docno': 'EntityId'})
        .to_csv(f'{args.o}/submit_results.csv', index=False)
    )