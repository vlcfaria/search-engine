from Experiment import Experiment
import pyterrier as pt
from helper.jsonlHandler import iter_jsonl, transform_fields
import xgboost as xgb

class BM25MART(Experiment):
    '''BM25 with LambdaMART'''
    def __init__(self, index_path: str, corpus_path: str= ''):
        super().__init__(index_path, corpus_path)

        #Make sure we have 3 fields on the index
        assert 3 == self.index.getCollectionStatistics().getNumberOfFields()
        
        self.name = 'bm25MART'

        search_field = lambda field,val: f"{field}({val})"

        #Title features
        title = pt.apply() >> pt.rewrite.tokenise()

        #Body features

        #Keyword features

        #Retrieve with BM25
        self.search_pipeline = pt.rewrite.tokenise() >> pt.terrier.Retriever(self.index, wmodel='BM25')

        # this configures xgb as LambdaMART
        params = {'objective': 'rank:ndcg', 
                'learning_rate': 0.1, 
                'gamma': 1.0, 'min_child_weight': 0.1,
                'max_depth': 6,
                'verbose': 2,
                'random_state': 42 
                }
        
        #Add to final pipeline
        lambaMART = pt.ltr.apply_learned_model(xgb.sklearn.XGBRanker(**params), form='ltr')
        self.search_pipeline = self.search_pipeline >> lambaMART

        #Train our lambdaMART, will fit on our pipeline features
        self.search_pipeline.fit()
            
    def get_index(self, index_path: str):
        return pt.IndexFactory.of(f"{index_path}/data.properties")
    
    def build_index(self, index_path: str, corpus_path: str):
        #Index fields
        return (pt.IterDictIndexer(index_path, meta={'docno': 20}, 
                            text_attrs=['TITLE', 'TEXT', 'KEYWORDS'], 
                            fields=True, blocks=True, threads=8)
                .index(iter_jsonl(corpus_path, transform_fields)))