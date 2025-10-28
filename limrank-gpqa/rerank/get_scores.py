from rank1 import rank1
from rankllama import RankLLaMA
from argparse import ArgumentParser
import json
import mteb
from tqdm import tqdm

def prepare_data():
    with open("data/gpqa_query/gpqa_original_queries.jsonl", "r") as f:
        data = [json.loads(line) for line in f]
    with open("data/retrieved/bm25s.jsonl", "r") as f:
        retrieved_data = [json.loads(line) for line in f]
    
    formatted = []    
    
    for d, retrieved in zip(data, retrieved_data):
        for ctx in retrieved["ctxs"]:
            formatted.append([d['query'], ctx["text"]])
    return formatted

def main(args):
    model_name = args.model_name.strip()
    if model_name in ["Qwen/Qwen2.5-7B", "jhu-clsp/rank1-7b", "songtingyu/limrank"]:    
        model = rank1(model_name_or_path=model_name, num_gpus=args.num_gpus, )
    elif model_name in ["orionweller/rankllama-7b-merged"]:
        model = RankLLaMA(num_gpus=args.num_gpus)
    elif model_name in ["jhu-clsp/FollowIR-7B"]:
        model = mteb.get_model(model_name)
    else:
        model = rank1(model_name_or_path=model_name, num_gpus=args.num_gpus, )
    
    model_name = model_name.split("/")[-1]
    data = prepare_data()
    
    if model_name in ["FollowIR-7B"]:
        scores = []
        batch_size = 8
        # batch process the data
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i:i+batch_size]
            batch_scores = model.predict(batch)
            scores.extend(batch_scores)
    else:
        scores = model.predict(data)
    
    with open("data/retrieved/bm25s.jsonl", "r") as f:
        retrieved_data = [json.loads(line) for line in f]
    index = 0
    for ctxs_dict in retrieved_data:
        ctxs = ctxs_dict["ctxs"]
        # sort the ctxs by their scores
        local_scores = scores[index:index+len(ctxs)]
        assert len(ctxs) == len(local_scores)
        # only sort the ctxs
        ctxs = [ctx for _, ctx in sorted(zip(local_scores, ctxs),key=lambda x: x[0], reverse=True)]
        ctxs_dict["ctxs"] = [{
            "text": ctx["text"],
            "score": score
        } for ctx, score in zip(ctxs, local_scores)]
        index += len(ctxs)
    with open(f"data/reranked/{model_name}_reranked.jsonl", "w") as f:
        for ctxs_dict in retrieved_data:
            f.write(json.dumps(ctxs_dict) + "\n")
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, help="The name of the model to evaluate.")
    parser.add_argument("--num_gpus", type=int, help="The number of GPUs to use for evaluation.", default=2)
    
    args = parser.parse_args()
    main(args)
