from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import ipdb
import jsonlines
import numpy as np

weight_to_scale = {
    "Critical": 4,
    "Major": 3,
    "Minor": 2,
    "Additional": 1
}


def compute_fulfillment_for_task_id_batch_id(annotations, task_id: int, generation_id: int) -> float:
    relevant_annotations = [s for s in annotations if s['task_id'] == task_id and s['generation_id'] == generation_id]

    sum_max_scores = 0
    sum_actual_scores = 0
    for annotation in relevant_annotations:
        sum_max_scores += 1
        sum_actual_scores += int(annotation['judge_rating'].startswith("Yes"))

    average_score = sum_actual_scores / sum_max_scores

    return average_score


def compute_score_for_task_id_batch_id(annotations, task_id: int, generation_id: int) -> float:
    relevant_annotations = [s for s in annotations if s['task_id'] == task_id and s['generation_id'] == generation_id]

    sum_max_scores = 0
    sum_actual_scores = 0
    for annotation in relevant_annotations:
        sum_max_scores += weight_to_scale[annotation['weight']]
        sum_actual_scores += weight_to_scale[annotation['weight']] * int(annotation['judge_rating'].startswith("Yes"))

    average_score = sum_actual_scores / sum_max_scores

    return average_score


def compute_score_for_task_id(annotations, task_id: int) -> float:
    relevant_annotations = [s for s in annotations if s['task_id'] == task_id]

    sum_max_scores = 0
    sum_actual_scores = 0
    for annotation in relevant_annotations:
        sum_max_scores += weight_to_scale[annotation['weight']]
        sum_actual_scores += weight_to_scale[annotation['weight']] * int(annotation['judge_rating'].startswith("Yes"))

    average_score = sum_actual_scores / sum_max_scores

    return average_score


def get_mean_at_k(score_list, k: int):
    # mean estimate of mean@16 with k samples
    scores = [np.mean(combo) for combo in combinations(score_list, k)]
    return round(np.mean(scores), 10)


def get_std_at_k(score_list, k: int):
    # std of the estimate of mean@16 with k samples
    scores = [np.mean(combo) for combo in combinations(score_list, k)]
    return round(np.std(scores), 10)


if __name__ == "__main__":
    filename = Path("./data/standard_format_30_Aug_cleaned_w_filepath_filtered_infer_gemini-2.5-flash_reasoning_True_files_1_web_0_seed_16_model_nvdev_openai_gpt-oss-120b_reasoning_True.jsonl")
    k = 1

    with jsonlines.open(filename) as f:
        raw_samples = sorted(list(f), key=lambda x: x['idx'])

    # assign generation id to raw_samples
    annotations = []
    for batch_idx, batch_start_idx in enumerate(range(0, len(raw_samples), 1162)):
        batch_raw_samples = raw_samples[batch_start_idx:batch_start_idx + 1162]
        for raw_sample in batch_raw_samples:
            raw_sample['generation_id'] = batch_idx
            annotations.append(raw_sample)

    # Compute average per domain
    print(f"Domains")
    domains = set(annotation['domain'] for annotation in annotations)
    batch_domain_scores = {domain: [] for domain in domains}
    for domain in domains:
        task_ids = set(annotation['task_id'] for annotation in annotations if annotation['domain'] == domain)
        for generation_id in range(16):
            sum_actual_score = 0
            num_annotations = 0
            for task_id in task_ids:
                sum_actual_score += compute_score_for_task_id_batch_id(
                    annotations=annotations, task_id=task_id, generation_id=generation_id
                )
                num_annotations += 1

            average_score = sum_actual_score / num_annotations
            batch_domain_scores[domain].append(average_score)

    # print average and std of domain-wise scores across 16 batches
    for domain, scores in batch_domain_scores.items():
        print(f"{domain}@{k}: {get_mean_at_k(scores, k)} ({get_std_at_k(scores, k)})")

    # print overall average
    all_domain_scores = [np.mean(scores) for scores in zip(*batch_domain_scores.values())]
    print(f"Overall@{k}: {get_mean_at_k(all_domain_scores, k)} ({get_std_at_k(all_domain_scores, k)})")

    # Compute average per criterion type
    print(f"=" * 20)
    print("Criterion Types")
    criterion_types = {"Extraction (recall)", "Reasoning", "Style"}
    batch_criterion_type_scores = {criterion_type: [] for criterion_type in criterion_types}
    for criterion_type in criterion_types:
        task_ids = set(annotation['task_id'] for annotation in annotations if criterion_type in annotation['criterion_type'])
        for generation_id in range(16):
            sum_actual_score = 0
            num_annotations = 0
            for task_id in task_ids:
                sum_actual_score += compute_fulfillment_for_task_id_batch_id(
                    annotations=annotations, task_id=task_id, generation_id=generation_id
                )
                num_annotations += 1

            average_score = sum_actual_score / num_annotations
            batch_criterion_type_scores[criterion_type].append(average_score)

    # print average and sd of criterion type-wise scores across 16 batches
    for criterion_type, scores in batch_criterion_type_scores.items():
        print(f"{criterion_type}@{k}: {get_mean_at_k(scores, k)} ({get_std_at_k(scores, k)})")

    ipdb.set_trace()
    pass