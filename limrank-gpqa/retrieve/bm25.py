import bm25s
import sys
import os
sys.path.append(".")
import pickle
from pathlib import Path
import json
from argparse import ArgumentParser

# Load gpqa corpus
def load_corpus():
    with open("data/gpqa_dd/gpqa_datastore_pool_dedupped.jsonl", "r") as f:
        combined_data = [json.loads(line) for line in f]
    return combined_data 

def index():
    if os.path.exists("indices/bm25s_index"):
        print("Index already exists")
        return
    corpus = load_corpus()
    corpus_text = [doc["text"] for doc in corpus]
    # Tokenize the corpus and only keep the ids (faster and saves memory)   
    corpus_tokens = bm25s.tokenize(texts=corpus_text, stopwords="en")
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(corpus_tokens)
    retriever.save(save_dir="./indices/bm25s_index", corpus=corpus)

def load_index(path: str):
    # Load the index from the specified path
    if not os.path.exists(path):
        index()
    retriever = bm25s.BM25.load(path, load_corpus=True)
    return retriever

# save the search results to a json file
def search(topk: int):
    with open("data/gpqa_query/gpqa_original_queries.jsonl", "r") as f:
        data = [json.loads(line) for line in f]
    retriever = load_index("./indices/bm25s_index")
    # search_dict = {}
    search_results = []
    queries = [d['query'] for d in data]
    queries_token_ids = bm25s.tokenize(texts=queries, stopwords="en")
    results, scores = retriever.retrieve(queries_token_ids, k=topk)
    for i, d in enumerate(queries):
        doc_list = []
        for j in range(results.shape[1]):
            doc, score = results[i, j], scores[i, j]
            doc_list.append({
                "text": doc["text"],
                "score" : str(score)
            })
            
        search_results.append({
            "ctxs" : doc_list
        })
    
    with open("data/retrieved/bm25s.jsonl", "w") as f:
        for item in search_results:
            f.write(json.dumps(item) + "\n")    
            
def main():
    # Search the corpus
    topk = 100
    search( topk=topk)
    
if __name__ == "__main__":
    main()
