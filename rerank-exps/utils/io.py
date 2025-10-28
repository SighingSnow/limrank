import json

def export_json(path: str, data):
    json.dump(data, open(path, "w"), indent=4, ensure_ascii=False)

def export_jsonl(path: str, data):
    with open(path, 'w') as f:
        for line in data:
            f.write(json.dumps(line) + "\n")

def read_json(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def read_jsonl(path: str):
    with open(path, "r") as f:
        data = f.readlines()
    data = [json.loads(line) for line in data]
    return data