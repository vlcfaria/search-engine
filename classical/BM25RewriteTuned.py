from Experiment import Experiment
import pyterrier as pt
from helper.jsonlHandler import iter_jsonl, transform_raw
import pandas as pd

class BM25(Experiment):
    '''BM25 with RM3 query rewriting, but available for tuning with `tune`, and with established tuning in the default pipeline'''

    def __init__(self, index_path: str, corpus_path: str= ''):
        super().__init__(index_path, corpus_path)
        
        self.name = 'bm25-rm3-tuned'
        bm25 = pt.terrier.Retriever(self.index, wmodel='BM25')
        rm3 = pt.rewrite.RM3(self.index)

        self.search_pipeline = pt.rewrite.tokenise() >> bm25 >> rm3 >> bm25
    
    def get_index(self, index_path: str):
        return pt.IndexFactory.of(f"{index_path}/data.properties")
    
    def build_index(self, index_path: str, corpus_path: str):
        #Index raw data
        return (pt.IterDictIndexer(index_path, meta={'docno': 20}, blocks=True, threads=8)
                    .index(iter_jsonl(corpus_path, transform_raw)))

    def tune(self, topics_path: str, qrels_path: str, out_dir: str = ''):
        '''Tunes RM3 and BM25 parameters'''

        qrels = pd.read_csv(qrels_path, 
                            dtype={'qid': 'object', 'docno': 'object', 'label': 'int64'})
        topics = pd.read_csv(topics_path, 
                             dtype={'qid': 'object', 'query': 'object'})

        bm25 =  pt.terrier.Retriever(self.index, wmodel="BM25", controls={"bm25.b" : 0.75, "bm25.k_1": 0.75, "bm25.k_3": 0.75})
        rm3 = pt.rewrite.RM3(self.index)

        #pipe = pt.rewrite.tokenise() >> bm25 >> rm3 >> bm25
        pipe = bm25

        param_map = {
            bm25: {"bm25.b"  : [0, 0.25, 0.5, 0.75, 1],
                "bm25.k_1": [0.3, 0.9, 1.4, 2],
                "bm25.k_3": [0.5, 2, 8, 12, 20]
            }
        }

        #TODO this is slow even with minimal parameters, setting jobs= doesnt work for me
        pipe = pt.GridSearch(pipe, param_map, topics, qrels, verbose=True, metric='ndcg')
        print(pipe)
        
        #Experriment on our tuned vs original!
        ans = pt.Experiment([pipe, self.search_pipeline], topics,  qrels, ['ndcg_cut', 'recall'], names=['tuned', 'original'])
        
        if out_dir:
            ans.to_csv(f'{out_dir}/{self.name}-tuning.csv')
        else:
            print(ans)