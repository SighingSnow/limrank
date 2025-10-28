import hashlib
import json

PROMPT_DICT = {
    "SciFact": """Claim: FILL_QUERY_HERE

A relevant passage would provide evidence that either **supports** or **refutes** this claim. A passage with any information on any related subpart should be relevant.""",

    "ClimateFEVER": """I am looking to write an essay for and need to find evidence the supports or contradict this statement:

FILL_QUERY_HERE

If the passage provides information that supports or contradicts it in any way it is relevant so I can cite it.

""",

    "TRECCOVID": """FILL_QUERY_HERE If the article answers any part of the question it is relevant.""",

    "ArguAna": """I am looking to write an essay and need to find counterarguments against this statement:

FILL_QUERY_HERE

Does this passage have any counterargument or evidence that could be used to help me?

""",

    "DBPedia": """I am looking to write an essay on this topic and need as much related background information to help me. The topic is:

FILL_QUERY_HERE

If the passage provides any background information that could be connected it is relevant.

""",

    "FiQA2018": """FILL_QUERY_HERE Find a passage that would be a good answer from StackExchange.""",

    "NFCorpus": """Topic: FILL_QUERY_HERE

Given the above topic, I need to learn about all aspects of it. It does not need to be directly relevant, only tangentially informational. Please mark as relevant any passages with even weak connections. I need to learn fast for my job, which means I need to understand each part individually.

Again remember, any connection means relevant even if indirect. So if it is not addressed, that is okay -- it does not need to be explicitly. 

Find me passages with any type of connection, including weak connections!!!!""",

    "Touche2020": """I am looking to write an essay and need to find arguments for or against this statement:

FILL_QUERY_HERE

Does this passage have any argument or evidence that could be used to help me?

""",

    "SCIDOCS": """papers that could be cited in FILL_QUERY_HERE. Anything with even indirect relevance should be relevant. This includes papers in the same broader field of science""",

    "BrightRetrieval_aops": """Find different but similar math problems to FILL_QUERY_HERE\n\nA document is relevant if it uses the same class of functions and shares **any** overlapping techniques.""",

    "BrightRetrieval_theoremqa_questions": """Find a passage which uses the same mathematical process as this one: FILL_QUERY_HERE""",

    "BrightRetrieval_leetcode": """I am looking to find different problems that share similar data structures (of any kind) or algorithms (e.g. DFS, DP, sorting, traversals, etc.). I am looking for problems that share one or both of these similarities to this:
    
FILL_QUERY_HERE

Does this passage share any similarities? e.g. if there was a textbook on leetcode problems, this would be in the same book even though it could be in a different chapter. 


""",

    "BrightRetrieval_pony": """I will use the programming language pony. Problem: FILL_QUERY_HERE

But to solve the problem above, I need to know things about pony. A passage is relevant if it contains docs that match **any** part (even basic parts) of the code I will have to write for the above program.""",

    "BrightRetrieval": """Can you find background information about the concepts used to answer the question:

FILL_QUERY_HERE

A passage is relevant if it contains background information about a **sub-concept** that someone might cite/link to when answering the above question."""

}

PROMPT_DICT["BrightRetrieval_theoremqa_theorems"] = PROMPT_DICT["BrightRetrieval_theoremqa_questions"]


def get_prompt(task_name, subtask_name: str = None):
    if subtask_name is not None and task_name in PROMPT_DICT:
        # if subtask is present, use that, otherwise use just the task name
        if f"{task_name}_{subtask_name}" in PROMPT_DICT:
            return PROMPT_DICT[f"{task_name}_{subtask_name}"]
        else: # default for subtask (e.g. BrightRetrieval)
            return PROMPT_DICT[task_name]
    elif task_name in PROMPT_DICT:
        # no subtask
        return PROMPT_DICT[task_name]
    else:
        return None


BEIR_DATASETS = [
    "ArguAna",
    "ClimateFEVER",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "TRECCOVID",
    "Touche2020",
]

FOLLOWIR_DATASETS = [
    "Core17",
    "Robust04",
    "News21"
]

def validate_json(file_path: str) -> bool:
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        # assert there are string keys and that within that a dict of key -> float values
        for key in data:
            assert isinstance(key, str), f"Key is not a string: {key}"
            assert isinstance(data[key], dict), f"Data is not a dict: {data[key]}"
            for inner_key, inner_value in data[key].items():
                assert isinstance(inner_key, str), f"Inner key is not a string: {inner_key}"
                assert isinstance(inner_value, float), f"Inner value is not a float: {inner_value}"
        return True
    except Exception as e:
        print(e)
        return False
