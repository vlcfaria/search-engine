import argparse
import json
from ragatouille import RAGPretrainedModel
import time

def get_corpus_batch(corpus_path, start=0, batch_size=10000):
    documents = []
    doc_ids = []
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for num_line, line in enumerate(f):
            if num_line < start:
                continue
            if num_line >= start + batch_size:
                break
            
            doc = json.loads(line.strip())
            text = f'{doc.get("title", "")} \n {doc.get("text", "")} \n Keywords: {", ".join(doc.get("keywords", []))}.'
            documents.append(text)
            doc_ids.append(doc.get('id', ''))
    
    return documents, doc_ids

class ColBERT:    
    def __init__(self, name: str = "ColBERT"):
        self.name = name
        self.index_path = None
        self.RAG = None

    def build_index(self, index_name: str, args):
        self.RAG = RAGPretrainedModel.from_pretrained(args.model_name)

        start_doc = 0
        batch_size = 5000000
        first_batch = True

        start_time = time.time()
        
        while True:
            print(f"Processing batch starting at document {start_doc}")
            
            document_batch, document_ids = get_corpus_batch(
                args.corpus, start=start_doc, batch_size=batch_size
            )
            
            if not document_batch:
                break
            
            if first_batch:
                index_path = self.RAG.index(
                    index_name=index_name, 
                    collection=document_batch,
                    document_ids=document_ids,
                    use_faiss=True,
                    overwrite_index=True
                )
                print(f"Initial index created at: {index_path}. Time taken: {time.time() - start_time:.2f} seconds.")
                self.index_path = index_path
                first_batch = False
            else:
                self.RAG.add_to_index(
                    new_collection=document_batch,
                    new_document_ids=document_ids,  
                )
            
            start_doc += len(document_batch)
            print(f"Indexed {start_doc} documents total. Time taken for this batch: {time.time() - start_time:.2f} seconds.")
        
        return self.index_path

    def search(self, query: str, k: int = 10):
        if self.RAG is None:
            raise ValueError("RAG model is not initialized. Please build the index first.")
        
        results = self.RAG.search(query, k=k)
        return results

def main():
    parser = argparse.ArgumentParser(description='ColBERT Indexing and Search')
    parser.add_argument('--corpus', type=str, default='./dataset/corpus.jsonl', help='Path to corpus file')
    parser.add_argument('--index_name', type=str, default='colbert_index', help='Name of the index')
    parser.add_argument('--model_name', type=str, default='colbert-ir/colbertv2.0', help='ColBERT model name')
    
    args = parser.parse_args()
    
    colbert = ColBERT()
    
    print(f"Building ColBERT index from corpus: {args.corpus}")
    index_path = colbert.build_index(args.index_name, args)
    print(f"ColBERT index created at: {index_path}")

if __name__ == "__main__":
    main()
