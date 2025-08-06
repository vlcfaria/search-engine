from ragatouille import RAGPretrainedModel
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Query ColBERT index')
    parser.add_argument('--index', required=True, help='Path to the ColBERT index')
    parser.add_argument('--queries', required=True, help='Path to the queries CSV file')
    parser.add_argument('--output', default='bm25-colbert.csv', help='Output CSV file path')
    
    args = parser.parse_args()
    
    RAG = RAGPretrainedModel.from_index(args.index)

    queries = pd.read_csv(args.queries)

    results = { 'QueryId': [], 'EntityId': [] }
    for index, row in queries.iterrows():
        query = row['query']
        query_id = row['qid']

        print(f"Processing query: {query}")
        
        results_batch = RAG.search(query, k=100)
        
        for result in results_batch:
            results['QueryId'].append(str(query_id).zfill(3))
            results['EntityId'].append(result['document_id'])

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
