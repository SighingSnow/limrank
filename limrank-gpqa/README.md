# LimRank-GPQA

We mainly conduct our experiments on the GPQA `Diamond` dataset based on [ReasonIR]() settings. 


## Data Preparation
```bash
cd limrank-gpqa
mkdir -p data data/gpqa_query data/gpqa_searched_results data/gpqa_dd
wget https://huggingface.co/datasets/rulins/gpqa_original_queries/resolve/main/gpqa_original_queries.jsonl -O data/gpqa_query/gpqa_original_queries.jsonl
wget https://huggingface.co/datasets/rulins/gpqa_searched_results_from_massiveds_non_cc/resolve/main/gpqa_original_queries_searched_results.jsonl -O data/gpqa_searched_results/gpqa_original_queries_searched_results.jsonl
wget https://huggingface.co/datasets/Idavidrein/gpqa/resolve/main/gpqa_diamond.csv -O data/gpqa_raw/gpqa_diamond.csv 
wget https://huggingface.co/datasets/songtingyu/gpqa_corpus_for_limrank/blob/main/gpqa_datastore_pool_dedupped.jsonl -O data/gpqa_dd/gpqa_datastore_pool_dedupped.jsonl
```

Then we need to construct the bm25 index for retrieving documents. It will automatically generated the data/retrieved/bm25s.jsonl for reranker to use. 
```bash
python retrieve/bm25.py
```

## Rerank & Evaluation

Use the following command to rerank the documents and save the reranked results to data/reranked/.
```bash
python rerank/get_scores.py --model_name [songtingyu/limrank-7b]  # let the model to rerank 
python eval/inference.py --model_name [songtingyu/limrank-7b] # do rag inference
python eval/openai_eval.py # use openai to judge the final answers correctness
```