from Experiment import Experiment
import pyterrier as pt
from helper.jsonlHandler import iter_jsonl, transform_fields
import xgboost as xgb
import numpy as np
import pandas as pd

class BM25MART(Experiment):
    '''BM25 with LambdaMART'''
    def __init__(self, index_path: str, topics_path: str, qrels_path: str, corpus_path: str= ''):
        super().__init__(index_path, corpus_path)

        #Make sure we have 3 fields on the index
        assert 3 == self.index.getCollectionStatistics().getNumberOfFields()

        self.docindex = self.index.getDocumentIndex()
        
        self.name = 'bm25MART'

        def search_field(field):
            return lambda q: f"{field}:({q['query']})"
        
        #Defining the initial retrieval process --
        initial_retrieval = pt.terrier.Retriever(self.index, wmodel='BM25', controls={"bm25.b" : 0.75, "bm25.k_1": 1.2, "bm25.k_3": 1.2})
                
        #Feature engineering --
        bm25_features = pt.terrier.Retriever(self.index, wmodel='BM25')
        title = pt.apply.query(search_field('TITLE')) >> bm25_features
        body = pt.apply.query(search_field('TEXT')) >> bm25_features
        keywords = pt.apply.query(search_field('KEYWORDS')) >> bm25_features

        dph = pt.terrier.Retriever(self.index, wmodel='DPH')
        pl2 = pt.terrier.Retriever(self.index, wmodel='PL2')
        dfr = pt.terrier.Retriever(self.index, wmodel='DFR_BM25')
        dlh = pt.terrier.Retriever(self.index, wmodel='DLH')

        sdm = pt.rewrite.SequentialDependence() >> bm25_features

        dph_title = pt.apply.query(search_field('TITLE')) >> dph
        dph_text = pt.apply.query(search_field('TEXT')) >> dph
        dph_keywords = pt.apply.query(search_field('KEYWORDS')) >> dph
        
        pl2_title = pt.apply.query(search_field('TITLE')) >> pl2
        pl2_text = pt.apply.query(search_field('TEXT')) >> pl2
        pl2_keywords = pt.apply.query(search_field('KEYWORDS')) >> pl2

        query_len = pt.apply.doc_score(lambda row: len(row['query_0']))
        #This resolves to the number of words (after stopword removal) of the 'text' field
        doc_len_text = pt.apply.doc_score(lambda row: self.docindex.getDocumentLength(row['docid']))

        features_pipe = (
            title ** body ** keywords ** #Fielded BM25
            dph ** pl2 ** #Other retrieval models
            query_len ** doc_len_text ** #Static features
            initial_retrieval #Also use initial retrieval signal
            ** dfr ** dlh ** # ADD NEW MODELS
            sdm# ADD PROXIMITY
            #** dph_title ** dph_text ** dph_keywords **
            #pl2_title ** pl2_text ** pl2_keywords
        )

        #Establishing LTR --
        #this configures xgb as LambdaMART
        lmart = xgb.sklearn.XGBRanker(objective='rank:ndcg',
            learning_rate=0.05,
            gamma=1.0,
            min_child_weight=0.1,
            n_estimators=200,
            max_depth=6,
            verbose=2,
            random_state=42)
        lambdaMART = pt.ltr.apply_learned_model(lmart, form='ltr')
        
        #Building the full pipeline -- 
        # Tokenize -> Initial retrieval -> Build features -> LambdaMART
        self.search_pipeline = pt.rewrite.tokenise() >> initial_retrieval >> features_pipe >> lambdaMART
        self.search_pipeline.compile()

        #Training -- 
        qrels = pd.read_csv(qrels_path, 
                            dtype={'qid': 'object', 'docno': 'object', 'relevance': 'int64'})
        topics = pd.read_csv(topics_path, 
                             dtype={'qid': 'object', 'query': 'object'})
        
        train_topics, valid_topics, test_topics = np.split( #shuffle topics
            topics.sample(frac=1, random_state=43),
            [int(.6*len(topics)), int(.8*len(topics))]
        )

        self.search_pipeline.fit(train_topics, qrels, valid_topics, qrels)
        print(lmart.feature_importances_)

        ans = pt.Experiment([self.search_pipeline], test_topics, qrels, 
                            eval_metrics=['recip_rank', 'ndcg_cut', 'recall'], 
                            names=[self.name])
        print(ans)
            
    def get_index(self, index_path: str):
        return pt.IndexFactory.of(f"{index_path}/data.properties")
    
    def build_index(self, index_path: str, corpus_path: str):
        #Index fields
        return (pt.IterDictIndexer(index_path, meta={'docno': 20}, 
                            text_attrs=['title', 'text', 'keywords'], 
                            fields=True, blocks=True, threads=8)
                .index(iter_jsonl(corpus_path, transform_fields)))


if __name__ == '__main__':
    s = BM25MART('./terrier_index_fields', './queries/train_queries.csv', './queries/train_qrels.csv', './dataset/corpus.jsonl')
    #!!! UPDATE index.direct.fields.names and index.inverted.fields.names TO UPPERCASE LATER (TITLE,TEXT,KEYWORDS)

    s.benchmark('./queries/train_queries.csv', './queries/train_qrels.csv')
    s.results_tests('./queries/test_queries.csv', './queries')