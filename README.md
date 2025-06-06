# search-engine

Use **Python 3.10** then
```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Usage

For reproducibility, the whole search pipeline (building index up to final results) logic should be stored in a single class, derived from the base class in [Experiment.py](./Experiment.py). For a concrete example, check out [BaseBM25.py](./BaseBM25.py) for a simple BM25 retrieval, including usage. For compatibility make sure to use Pyterrier indexes and transformers.

You can test out the efficiency of a given model using the `benchmark` method with `train_qrels.csv` and `train_queries.csv`, and then produce the kaggle submission with `results_tests`.