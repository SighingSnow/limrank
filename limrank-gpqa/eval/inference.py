from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json
from argparse import ArgumentParser

def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Configurae the sampling parameters (for thinking mode)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, top_k=20, max_tokens=8192)

    # Initialize the vLLM engine
    llm = LLM(
        model=model_name, 
        gpu_memory_utilization=0.9, 
        tensor_parallel_size=2,
        trust_remote_code=True,
        max_model_len=8192,
        dtype="float16"
    )
    return (llm, sampling_params), tokenizer

def inference(model_name, topk: int = 10, reranker: str = "rank1-7b"):
    with open("data/gpqa_query/gpqa_original_queries.jsonl", "r") as f:
        data = [json.loads(line) for line in f]
    if topk > 0:
        with open(f"data/{reranker}_retrieved.jsonl", "r") as f:
            reranked_data = [json.loads(line) for line in f]
    
    (llm, sampling_params), tokenizer = load_model(model_name=model_name)
    messages = []
    if topk > 0:
        for item, ctx in zip(data, reranked_data):
            item["ctxs"] = ctx["ctxs"][:topk]
            item["query"] = item["query"] + "\nHere are some relevant documents:\n"
            for i, c in enumerate(item["ctxs"]):
                item["query"] += f"Document {i + 1}: {c['text']}\n"
            item["query"] += "Answer the question based on the documents above."
            message = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{item['query']}"}
            ]
            messages.append(message)        
    else:
        messages = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{item['query']}"}
            ]
            for item in data
        ]
    
    texts = tokenizer.apply_chat_template(messages,tokenize=False, add_generation_prompt=True)
    outputs = llm.generate(texts, sampling_params=sampling_params)

    for q, o in zip(data, outputs):
        q["output"] = o.outputs[0].text.strip()
    
    model_name = model_name.split("/")[-1]
    if topk > 0:
        with open(f"data/{model_name}_outputs_top{topk}_rerank{reranker}.json", "w") as f:
            json.dump(data, f, indent=4)
    else:    
        with open(f"data/{model_name}_outputs.json", "w") as f:
            json.dump(data, f, indent=4)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B", help="The name of the model to use.")
    parser.add_argument("--topk", type=int, default=10, help="The number of top retrieved documents to use.")
    parser.add_argument("--reranker", type=str, default="rank1-7b", help="The reranker to use for retrieval.")
    
    args = parser.parse_args()
    inference(args.model_name, args.topk, args.reranker)