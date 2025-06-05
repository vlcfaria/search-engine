# search-engine

Use **Python 3.10** then
```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## For indexing

### Index using pyserini (more standard)

Convert corpus first:

```bash
$ mkdir collection
$ python convert.py
$ mv converted.jsonl collection
```

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

### Index using Pyterrier (more customizable)

Using standard (non-converted) corpus

```bash
$ python index_pyterrier.py -i <path to corpus> -o <output directory>
```

Run experiment with PyTerrier:

```bash
python benchmark.py
```