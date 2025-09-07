import json
from collections import Counter, defaultdict

import jsonlines
import numpy as np
import sys
from functools import cache 
from itertools import combinations




weight_to_scale = {
    "Critical": 4,
    "Major": 3,
    "Minor": 2,
    "Additional": 1
}


def get_predicted_score_per_task_id_e2e(condition=None, value=None, data=None):
    task_id_model_to_max_score = defaultdict(int)
    task_id_model_to_achieved_score = defaultdict(int)
    task_id_to_domain =  defaultdict(str)

    criterion_type_to_fulfilment = defaultdict(list)

    for i in data:
        weight = i["weight"]
        scale = weight_to_scale[weight]
        task_id = str(i["task_id"])
        model = i["model"]
        domain = i["domain"]
        criterion_types = i["criterion_type"]

        task_id_model_to_max_score[(task_id, model)] += scale
        task_id_to_domain[task_id] = domain

        criterion_fulfilment = (i[condition].startswith(value) if isinstance(value, str) else i[condition] == value)
        
        if criterion_fulfilment: 
            task_id_model_to_achieved_score[(task_id, model)] += scale

        for criterion_type in criterion_types:
            criterion_type_to_fulfilment[criterion_type].append(int(criterion_fulfilment))

    task_id_to_scores = defaultdict(float)

    for task_id_model in task_id_model_to_max_score:
        task_id, model = task_id_model
        score = task_id_model_to_achieved_score[task_id_model] / task_id_model_to_max_score[task_id_model]
        task_id_to_scores[task_id] = score

    domain_to_scores = defaultdict(list)

    for task_id, adjusted_score in task_id_to_scores.items():
        domain = task_id_to_domain[task_id]
        domain_to_scores[domain].append(adjusted_score)
    
    domain_average = {domain: round(np.mean(domain_to_scores[domain]), 20) for domain in domain_to_scores}
    
    all_domains = round(np.mean(list(domain_average.values())), 20)

    # add overall average to all_domains
    domain_average['Overall'] = all_domains

    # add average of each criterion type to all_domains
    for criterion_type in criterion_type_to_fulfilment:
        domain_average[criterion_type] = round(np.mean(criterion_type_to_fulfilment[criterion_type]), 20)
    
    for key in domain_average:
        domain_average[key] = round(domain_average[key]*100, 20)
    
    prompt_tokens = list({i["task_id"]: i["prompt_tokens"] for i in data if isinstance(i["prompt_tokens"], int)}.values())

    completion_tokens = list({i["task_id"]:i["completion_tokens"] for i in data if isinstance(i["completion_tokens"], int)}.values())
    
    domain_average["prompt_tokens"] = round(np.mean(prompt_tokens))
    domain_average["completion_tokens"] = round(np.mean(completion_tokens))

    judge_prompt_tokens = [i["judge_prompt_tokens"] for i in data]
    judge_completion_tokens = [i["judge_completion_tokens"] for i in data]

    domain_average["judge_prompt"]= round(np.mean(judge_prompt_tokens))
    domain_average["judge_completion"]= round(np.mean(judge_completion_tokens))

    return domain_average


def get_pass_at_k(score_list, k: int):
    scores = [np.mean(combo) for combo in combinations(score_list, k)]
    return round(np.mean(scores), 20)


def get_std_at_k(score_list, k: int):
    scores = [np.mean(combo) for combo in combinations(score_list, k)]
    return round(np.std(scores), 20)
        

if __name__ == "__main__":
    filename = "data/standard_format_30_Aug_cleaned_w_filepath_filtered_infer_gemini-2.5-flash_reasoning_True_files_1_web_0_seed_16_model_nvdev_openai_gpt-oss-120b_reasoning_True.jsonl"
    K = 16

    with jsonlines.open(filename) as f:
        data = list(f)

    data.sort(key=lambda x: x['idx'])

    assert len(data) % 1162 == 0, f"expect multiples of 1162 samples only have {len(data)}"
    # total-wise average
    score_by_subdomain = defaultdict(list)

    for i in range(0, len(data), 1162):
        subdata = data[i:i+1162]
        judge_rated_model_performance = get_predicted_score_per_task_id_e2e(condition="judge_rating", value="Yes", data=subdata)
        # print("Score:" ,judge_rated_model_performance)
        for key in judge_rated_model_performance:
            score_by_subdomain[key].append(judge_rated_model_performance[key])

    keys = ['Physics PhD', 'Chemistry PhD', 'Investment Services', 'Consulting', 'Overall', 'Extraction (recall)', 'Reasoning', 'Style', 'prompt_tokens', 'completion_tokens']
    for key_idx, key in enumerate(keys):
        print(f"Key: {round(get_pass_at_k(score_by_subdomain[key], K), 10)} ({round(get_std_at_k(score_by_subdomain[key], K), 10)}")