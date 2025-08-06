import pyterrier as pt
from helper.jsonlHandler import iter_jsonl, transform_raw

class BM25():
    '''Base BM25, only applying standard Terrier tokenization/stemming/stopwords and only using BM25 for ranking'''
    def __init__(self, index_path: str, corpus_path: str= ''):        
        self.name = 'base-bm25'

        if corpus_path:
            self.index = self.build_index(index_path, corpus_path)
        else:
            self.index = self.get_index(index_path)

        self.search_pipeline = pt.rewrite.tokenise() >> pt.terrier.Retriever(self.index, wmodel='BM25')
    
    def get_index(self, index_path: str):
        return pt.IndexFactory.of(f"{index_path}/data.properties")
    
    def build_index(self, index_path: str, corpus_path: str):
        #Index raw data
        return (pt.IterDictIndexer(index_path, meta={'docno': 20, 'text': 2048}, blocks=True, threads=8)
                    .index(iter_jsonl(corpus_path, transform_raw)))

#Example usage
if __name__ == '__main__':
    #Make an index at `./terrier_index` using ./dataset/corpus.jsonl
    bm = BM25('./terrier_index', './dataset/corpus.jsonl')