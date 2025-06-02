# search-engine

Use **Python 3.10** then
```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## For indexing

Convert corpus first:

```bash
$ mkdir collection
$ python convert.py
$ mv converted.jsonl collection
```

Index using pyserini
```bash
$ mkdir index
$ python -m pyserini.index.lucene \
--collection JsonCollection \
--input collection \
--index index \
--generator DefaultLuceneDocumentGenerator \
--threads 8 \
--storePositions -storeDocvectors
```

Run experiment with PyTerrier:

```bash
python benchmark.py
```