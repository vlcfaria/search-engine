import json
with open('corpus.jsonl', 'r') as input:
    with open('converted.jsonl', 'w') as output:
        for line in input:
            obj = json.loads(line)
            content = ' \\n '.join([obj["title"], obj["text"], *obj["keywords"]])
            new = {'id': obj["id"], 'contents': content}
            output.write(f'{json.dumps(new)}\n')