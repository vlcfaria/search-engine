import pyterrier as pt
from pyterrier_anserini import AnseriniIndex
import pandas as pd

pt.java.init()
tokenizer = pt.autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()
def strip_markup(text):
    return " ".join(tokenizer.getTokens(text))

#Read qrels
qrels = pd.read_csv("./queries/train_qrels.csv", dtype={'qid': 'object', 'docno': 'object', 'label': 'int64'})
topics = pd.read_csv("./queries/train_queries.csv", dtype={'qid': 'object', 'query': 'object'})

#Benchmark model here
index = AnseriniIndex('./index')
bm25 = index.retriever("BM25")
#index = pt.IndexFactory.of("./terrier_index/data.properties")
#bm25 = pt.terrier.Retriever(index, wmodel="BM25")

ans = pt.Experiment([bm25], topics, qrels, eval_metrics=['recip_rank', 'ndcg'], names=['bm25'])
print(ans)

#print(bm25.search('information retrieval'))