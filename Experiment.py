import pandas as pd
import pyterrier as pt

class Experiment():
    def __init__(self, index_path: str, corpus_path: str = '') -> None:
        self.name = 'abstract'

        if corpus_path:
            self.index = self.build_index(index_path, corpus_path)
        else:
            self.index = self.get_index(index_path)

        self.search_pipeline = None
        
    def get_index(self, index_path):
        'Gets the given index from `index_path`'
        raise NotImplementedError()

    def build_index(self, index_path, corpus_path):
        'Builds the given index in `index_path`, using the given corpus'
        raise NotImplementedError()
    
    def benchmark(self, topics_path: str, qrels_path: str, out_dir: str = ''):
        'Benchmarks the established search pipeline, according to established metrics'

        qrels = pd.read_csv(qrels_path, 
                            dtype={'qid': 'object', 'docno': 'object', 'label': 'int64'})
        topics = pd.read_csv(topics_path, 
                             dtype={'qid': 'object', 'query': 'object'})
        
        ans = pt.Experiment([self.search_pipeline], topics, qrels, 
                            eval_metrics=['recip_rank', 'ndcg_cut', 'recall'], 
                            names=[self.name])
        
        if out_dir:
            ans.to_csv(f'{out_dir}/{self.name}-benchmark.csv')
        else:
            print(ans)

    def results_tests(self, test_query_path: str, out_dir: str):
        'Outputs the query results (up to 100) for each test query, using the search pipeline'

        #Load queries
        queries = pd.read_csv(test_query_path)

        #Transform applies the search
        df_ans = self.search_pipeline.transform(queries)

        #Detailed result
        df_ans.to_csv(f'{out_dir}/{self.name}-test-results.csv', index=False)

        #Formatted result, for submitting to kaggle
        ( df_ans[['qid', 'docno']]
            .rename(columns={'qid': 'QueryId', 'docno': 'EntityId'})
            .to_csv(f'{out_dir}/{self.name}-submit-results.csv', index=False)
        )