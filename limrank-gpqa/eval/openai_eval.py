import re
import json
import numpy as np
from collections import Counter
import string
import os, time
from collections import defaultdict
from math_equivalence import is_equiv
from openai_utils import openai_pipeline
from dotenv import load_dotenv
def compare_answers(outputs, golden_answers, golden_indices):
    load_dotenv('./.env')
    messages = [
        [
            {"role": "user", "content": f"Given the outputs: {output} and the golden answers: {answer}, please compare them. And the correct choice is {index}. If the outputs match the golden answer or the correct choice, return 'yes', otherwise return 'no'. Please only return 'yes' or 'no'."}
        ]
        for output, answer, index in zip(outputs, golden_answers, golden_indices)
    ]
    responses = openai_pipeline(messages=messages, engine_name="gpt-4.1-mini", temperature=0.0, max_tokens=1024, top_p=1.0)
    responses = [response.lower().strip() for response in responses]
    return responses

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate GPQA model outputs.")
    parser.add_argument("--dataset_name", type=str, default="gpqa", help="Name of the dataset.")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-7B", help="Path to the model output file.")
    parser.add_argument("--reranker", type=str, default="rank1-7b", help="Name of the reranker used.")
    parser.add_argument("--topk", type=int, default=5, help="Number of top documents to consider.")
    
    args = parser.parse_args()
    if args.topk > 0:
        output_path = f"data/{args.model_name}_outputs_top{args.topk}_rerank{args.reranker}.json"
    else:
        output_path = f"data/{args.model_name}_outputs.json"
    
    dataset_name = args.dataset_name
    # TODO: load model output file
    with open(output_path, "r") as f:
        data = json.load(f)

    with open("data/gpqa_query/diamond.json", "r") as f:
        gold_data = json.load(f)
        
    overall_metrics = {"acc" : 0.0}
    
    outputs = [item['output'] for item in data]
    golden_answers = [item['Correct Answer'] for item in gold_data]
    golden_indices = [item['Correct Choice'] for item in gold_data]
    responses = compare_answers(outputs, golden_answers, golden_indices)
    acc = sum(1 for response in responses if response == 'yes') / len(responses)
    overall_metrics["acc"] = acc
    
    # save the responses to data
    for i, item in enumerate(data):
        item['response'] = responses[i]
    
    os.makedirs("outputs/gpqa", exist_ok=True)
    if args.topk > 0:
        with open(f"outputs/gpqa/{args.model_name}_{args.reranker}_top{args.topk}_eval.json", "w") as f:
            json.dump(overall_metrics, f, indent=4)
        with open(f"outputs/gpqa/{args.model_name}_{args.reranker}_top{args.topk}_responses.json", "w") as f:
            json.dump(data, f, indent=4)
    else:
        with open(f"outputs/gpqa/{args.model_name}_eval.json", "w") as f:
            json.dump(overall_metrics, f, indent=4)
        with open(f"outputs/gpqa/{args.model_name}_responses.json", "w") as f:
            json.dump(data, f, indent=4)
            
