import json

def iter_jsonl(filename: str, transform):
    'Index an entity jsonl file, calling `transform` for every loaded json entity'

    with open(filename, 'rt') as file:
        for l in file:
            raw = json.loads(l)

            yield transform(raw)

def transform_raw(raw: dict[str, str]) -> dict[str,str]:
    'Transform an entity dictionary into another entity with a single text field'

    text = ' \n '.join([raw['title'], raw['text'], ' \n '.join(raw['keywords'])])
    return {'docno': raw['id'], 'text': text}

def transform_fields(raw: dict[str, str]) -> dict[str,str]:
    'Transform an entity dictionary into an entity with multiple fields'

    return {'docno': raw['id'],
            'TITLE': raw['title'],
            'TEXT': raw['text'], 
            'KEYWORDS': ' \n '.join(raw['keywords'])}
