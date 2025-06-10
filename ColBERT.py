import argparse
import json
from ragatouille import RAGPretrainedModel

from Experiment import Experiment

MAX_DOCS = None

def get_corpus_text(corpus_path):
    """Generator to yield document text from corpus"""
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for num_line, line in enumerate(f):
            if MAX_DOCS is not None and num_line >= MAX_DOCS:
                break
            doc = json.loads(line.strip())
            yield f'{doc.get('title', '')}\n{doc.get('text', '')}\nKeywords: {', '.join(doc.get('keywords', []))}.'

def get_corpus_ids(corpus_path):
    """Generator to yield document IDs from corpus"""
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for num_line, line in enumerate(f):
            if MAX_DOCS is not None and num_line >= MAX_DOCS:
                break
            doc = json.loads(line.strip())
            yield doc.get('id', '')

class ColBERT(Experiment):
    """
    ColBERT model for RAG (Retrieval-Augmented Generation).
    This class initializes the ColBERT model, indexes documents, and prepares it for retrieval tasks.
    """
    
    def __init__(self, name: str = "ColBERT"):
        #super().__init__(name)
        self.name = name
        self.index_path = None

    def build_index(self, index_name: str, args):
        """Build ColBERT index from corpus documents"""
        RAG = RAGPretrainedModel.from_pretrained(args.model_name)
        
        # Get documents from corpus
        documents = list(get_corpus_text(args.corpus))
        
        print(documents[0])
        
        document_ids = list(get_corpus_ids(args.corpus))
        
        # Build index
        index_path = RAG.index(
            index_name=index_name, 
            collection=documents,
            document_ids=document_ids,
        )
        
        self.index_path = index_path
        return index_path

    def search(self, query: str, k: int = 10):
        """Search using ColBERT index"""
        if not self.index_path:
            raise ValueError("Index not built. Call build_index first.")
        
        # Load the index for searching
        RAG = RAGPretrainedModel.from_index(self.index_path)
        results = RAG.search(query, k=k)
        
        return results

def main():
    parser = argparse.ArgumentParser(description='ColBERT Indexing and Search')
    parser.add_argument('--corpus', type=str, default='./dataset/corpus.jsonl', help='Path to corpus file')
    parser.add_argument('--index_name', type=str, default='colbert_index', help='Name of the index')
    parser.add_argument('--model_name', type=str, default='colbert-ir/colbertv2.0', help='ColBERT model name')
    parser.add_argument('--overwrite_index', action='store_true', help='Overwrite existing index')
    
    args = parser.parse_args()
    
    # Initialize ColBERT
    colbert = ColBERT()
    
    # Build index
    print(f"Building ColBERT index from corpus: {args.corpus}")
    index_path = colbert.build_index(args.index_name, args)
    print(f"ColBERT index created at: {index_path}")

if __name__ == "__main__":
    main()