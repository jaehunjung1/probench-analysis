import random
from collections import defaultdict
from pathlib import Path

import ipdb
import jsonlines
import numpy as np
from tqdm import tqdm

from modules.optimal_allocation import get_optimal_num_response_allocation
from zhilin import get_predicted_score_per_task_id_e2e


if __name__ == "__main__":
    # parameters
    filename_dict = {
        "gemini-2.5-flash_reasoning_True": Path("data/standard_format_30_Aug_cleaned_w_filepath_filtered_infer_"
                                                "gemini-2.5-flash_reasoning_True_files_1_web_0_seed_16_model_"
                                                "nvdev_openai_gpt-oss-120b_reasoning_True.jsonl"),
        "gemini-2.5-pro_reasoning_True": Path("data/standard_format_30_Aug_cleaned_w_filepath_filtered_infer_"
                                              "gemini-2.5-pro_reasoning_True_files_1_web_0_seed_16_model_"
                                              "nvdev_openai_gpt-oss-120b_reasoning_True.jsonl"),
        "o3_reasoning_medium": Path("data/standard_format_30_Aug_cleaned_w_filepath_filtered_infer_"
                                    "o3_reasoning_medium_files_1_web_0_seed_16_model_"
                                    "nvdev_openai_gpt-oss-120b_reasoning_True.jsonl"),
        "o4-mini_reasoning_medium": Path("data/standard_format_30_Aug_cleaned_w_filepath_filtered_infer_"
                                         "o4-mini_reasoning_medium_files_1_web_0_seed_16_model_"
                                         "nvdev_openai_gpt-oss-120b_reasoning_True.jsonl"),
    }
    population_performance_dict = {
        "gemini-2.5-flash_reasoning_True": 0.576,
        "gemini-2.5-pro_reasoning_True": 0.603,
        "o3_reasoning_medium": 0.614,
        "o4-mini_reasoning_medium": 0.582,
    }

    total_num_responses = 160

    # === aggregate data === #
    # {model_name: {task_id: [score0, ..., score15], ...}, ...}
    model_to_task_id_to_scores = defaultdict(lambda: defaultdict(list))
    # {task_id: {model_name: [score0, ..., score15], ...}, ...}
    task_id_to_model_to_scores = defaultdict(lambda: defaultdict(list))
    for model_name, filename in filename_dict.items():
        with jsonlines.open(filename) as f:
            data = list(f)
        data.sort(key=lambda x: x['idx'])

        for batch_start_idx in range(0, len(data), 1162):
            batch = data[batch_start_idx:batch_start_idx + 1162]
            batch_performance, task_id_to_scores = get_predicted_score_per_task_id_e2e(
                condition="judge_rating", value="Yes", data=batch
            )

            for task_id, score in task_id_to_scores.items():
                model_to_task_id_to_scores[model_name][int(task_id)].append(score)
                task_id_to_model_to_scores[int(task_id)][model_name].append(score)

    # === compute task-level std across models === #
    task_id_to_model_to_average_var = {
        task_id: {model_name: float(np.var(task_id_to_model_to_scores[task_id][model_name]))
                  for model_name in task_id_to_model_to_scores[task_id].keys()}
        for task_id in task_id_to_model_to_scores.keys()
    }
    task_id_to_average_std = {
        task_id: float(np.sqrt(np.mean(list(task_id_to_model_to_average_var[task_id].values()))))
        for task_id in task_id_to_model_to_average_var.keys()
    }  # {task_id: average std}

    task_id_list = sorted(list(task_id_to_average_std.keys()))
    task_var_list = [float(task_id_to_average_std[task_id]) for task_id in task_id_list]

    # # === compute performance with dynamic allocation v2 === #
    # add_ratio = 0.5  # ratio of samples to add generations at each round
    # target_sample_sem = 0.02  # standard error of mean
    # num_min_samples = 1
    # for model_name, task_id_to_scores in model_to_task_id_to_scores.items():
    #     runs_to_task_scores = np.zeros((4096, len(task_id_list)))
    #     runs_to_num_generated = np.zeros((4096,))
    #     for run_idx in tqdm(range(4096), desc="Dynamic Allocation v2"):
    #         num_samples_to_add_in_current_round = len(task_id_list)
    #         samples_to_add = task_id_list
    #         while num_samples_to_add_in_current_round % 2 == 0:
    #             for task_id in enumerate(samples_to_add):
    #                 task_idx = samples_to_add
    #
    #         run_num_generated = 0
    #         for task_idx, task_id in enumerate(task_id_list):
    #             num_generated = 2
    #             sampled_scores = random.sample(task_id_to_scores[task_id], 2)
    #             while num_generated <= len(task_id_to_scores[task_id]):
    #                 num_generated += num_min_samples
    #                 sampled_scores += random.sample(task_id_to_scores[task_id], num_min_samples)
    #
    #                 sample_sem = np.std(sampled_scores) / np.sqrt(len(sampled_scores))
    #
    #                 if sample_sem <= target_sample_sem:
    #                     break
    #
    #             # exclude min / max
    #             if excluded_score := [s for s in sampled_scores if min(sampled_scores) < s < max(sampled_scores)]:
    #                 sampled_scores = excluded_score
    #             runs_to_task_scores[run_idx, task_idx] = np.mean(sampled_scores, axis=0)
    #             run_num_generated += num_generated
    #
    #         runs_to_num_generated[run_idx] = run_num_generated
    #
    #     average_num_generated = np.mean(runs_to_num_generated)
    #     overall_performance = np.mean(runs_to_task_scores, axis=1)  # (4096, 1)
    #     overall_std = np.sqrt(
    #         np.sum((overall_performance - population_performance_dict[model_name]) ** 2) / len(overall_performance))
    #     print(f"{model_name} - Dynamic allocation v2 (B={average_num_generated} / target_sample_std: {target_sample_sem}) "
    #           f"std: {np.std(overall_performance)}")
    #
    # print()

    # # === compute performance with dynamic allocation v1 === #
    # target_sample_sem = 0.02  # standard error of mean
    # num_min_samples = 2
    # for model_name, task_id_to_scores in model_to_task_id_to_scores.items():
    #     runs_to_task_scores = np.zeros((4096, len(task_id_list)))
    #     runs_to_num_generated = np.zeros((4096,))
    #     for run_idx in tqdm(range(4096), desc="Dynamic Allocation v1"):
    #         run_num_generated = 0
    #         for task_idx, task_id in enumerate(task_id_list):
    #             num_generated = 2
    #             sampled_scores = random.sample(task_id_to_scores[task_id], 2)
    #             while num_generated <= len(task_id_to_scores[task_id]):
    #                 num_generated += num_min_samples
    #                 sampled_scores += random.sample(task_id_to_scores[task_id], num_min_samples)
    #
    #                 sample_sem = np.std(sampled_scores) / np.sqrt(len(sampled_scores))
    #
    #                 if sample_sem <= target_sample_sem:
    #                     break
    #
    #             # exclude min / max
    #             if excluded_score := [s for s in sampled_scores if min(sampled_scores) < s < max(sampled_scores)]:
    #                 sampled_scores = excluded_score
    #             runs_to_task_scores[run_idx, task_idx] = np.mean(sampled_scores, axis=0)
    #             run_num_generated += num_generated
    #
    #         runs_to_num_generated[run_idx] = run_num_generated
    #
    #     average_num_generated = np.mean(runs_to_num_generated)
    #     overall_performance = np.mean(runs_to_task_scores, axis=1) # (4096, 1)
    #     overall_std = np.sqrt(np.sum((overall_performance - population_performance_dict[model_name]) ** 2)  / len(overall_performance))
    #     print(f"{model_name} - Dynamic allocation v1 (B={average_num_generated} / target_sample_std: {target_sample_sem}) "
    #           f"std: {np.std(overall_performance)}")
    #
    # print()


    # === compute performance with optimal allocation per model === #
    allocation, min_var = get_optimal_num_response_allocation(task_var_list, total_num_responses=total_num_responses)
    task_id_allocation = {task_id: int(alloc) for task_id, alloc in zip(task_id_list, allocation)}
    ipdb.set_trace()
    for model_name, task_id_to_scores in model_to_task_id_to_scores.items():
        runs_to_task_scores = np.zeros((4096, len(task_id_list)))
        for run_idx in range(4096):
            for task_idx, task_id in enumerate(task_id_list):
                allocation_for_this_task = allocation[task_idx]
                sampled_scores = [random.choice(task_id_to_scores[task_id]) for _ in range(allocation_for_this_task)]

                average_task_scores = np.mean(sampled_scores, axis=0)
                runs_to_task_scores[run_idx, task_idx] = average_task_scores

        overall_performance = np.mean(runs_to_task_scores, axis=1)
        overall_std = np.sqrt(np.sum((overall_performance - population_performance_dict[model_name]) ** 2)  / len(overall_performance))
        print(f"{model_name} - Optimal allocation (B={total_num_responses}) std: {np.std(overall_performance)}")

    print()

    # === compute performance with uniform allocation per model === #
    allocation = [int(total_num_responses / len(task_id_list))] * len(task_id_list)
    for model_name, task_id_to_scores in model_to_task_id_to_scores.items():
        runs_to_task_scores = np.zeros((4096, len(task_id_list)))
        for run_idx in range(4096):
            for task_idx, task_id in enumerate(task_id_list):
                allocation_for_this_task = allocation[task_idx]
                sampled_scores = [random.choice(task_id_to_scores[task_id]) for _ in range(allocation_for_this_task)]

                average_task_scores = np.mean(sampled_scores, axis=0)
                runs_to_task_scores[run_idx, task_idx] = average_task_scores

        overall_performance = np.mean(runs_to_task_scores, axis=1)
        overall_std = np.sqrt(np.sum((overall_performance - population_performance_dict[model_name]) ** 2)  / len(overall_performance))
        print(f"{model_name} - Uniform allocation (B={total_num_responses}) std: {np.std(overall_performance)}")

    print()

    # === compute performance with uniform allocation per model === #
    allocation = [int(total_num_responses / len(task_id_list))] * len(task_id_list)
    for model_name, task_id_to_scores in model_to_task_id_to_scores.items():
        runs_to_task_scores = np.zeros((4096, len(task_id_list)))
        for run_idx in range(4096):
            for task_idx, task_id in enumerate(task_id_list):
                allocation_for_this_task = allocation[task_idx]
                sampled_scores = [random.choice(task_id_to_scores[task_id]) for _ in range(allocation_for_this_task)]

                average_task_scores = np.median(sampled_scores, axis=0)
                runs_to_task_scores[run_idx, task_idx] = average_task_scores

        overall_performance = np.mean(runs_to_task_scores, axis=1)
        overall_std = np.sqrt(np.sum((overall_performance - population_performance_dict[model_name]) ** 2)  / len(overall_performance))
        print(f"{model_name} - Uniform median (B={total_num_responses}) std: {np.std(overall_performance)}")

    print()

    # === compute performance with uniform + heuristic min/max exclusion per model === #
    allocation = [int(total_num_responses / len(task_id_list))] * len(task_id_list)
    for model_name, task_id_to_scores in model_to_task_id_to_scores.items():
        runs_to_task_scores = np.zeros((4096, len(task_id_list)))
        for run_idx in range(4096):
            for task_idx, task_id in enumerate(task_id_list):
                allocation_for_this_task = allocation[task_idx]
                sampled_scores = [random.choice(task_id_to_scores[task_id]) for _ in range(allocation_for_this_task)]

                if min_max_removed := [s for s in sampled_scores if min(sampled_scores) < s < max(sampled_scores)]:
                    sampled_scores = min_max_removed

                average_task_scores = np.mean(sampled_scores, axis=0)
                runs_to_task_scores[run_idx, task_idx] = average_task_scores

        overall_performance = np.mean(runs_to_task_scores, axis=1)
        overall_std = np.sqrt(np.sum((overall_performance - population_performance_dict[model_name]) ** 2)  / len(overall_performance))
        print(f"{model_name} - Uniform + min / max exclusion (B={total_num_responses}) std: {np.std(overall_performance)}")


    # === compute std with brute-force independent runs as estimation === #
    for model_name, task_id_to_scores in model_to_task_id_to_scores.items():
        runs_to_task_scores = np.zeros((16, len(task_id_list)))  # (16, 40)
        for task_idx, task_id in enumerate(task_id_list):
            runs_to_task_scores[:, task_idx] = task_id_to_scores[task_id]

        overall_performance = np.mean(runs_to_task_scores, axis=1)  # (16, 1)
        overall_std = np.sqrt(np.sum((overall_performance - population_performance_dict[model_name]) ** 2)  / len(overall_performance))
        print(f"{model_name} - Brute-force std: {overall_std}")

    # # === analyze which rubric got more allocation and less === #
    # allocation, min_var = get_optimal_num_response_allocation(task_var_list, total_num_responses=total_num_responses)
    # task_id_to_allocation = {task_id: num_allocation for task_id, num_allocation in zip(task_id_list, allocation)}
    # with jsonlines.open("./data/probench_first_40_public_last_40_private.jsonl") as f:
    #     rubric_samples = list(f)
    #     rubrics_dict = {}  # task_id: rubric sample
    #     num_rubrics_dict = {}  # task_id: number of rubrics
    #     for rubric_sample in rubric_samples:
    #         rubrics_dict[int(rubric_sample['task_id'])] = rubric_sample
    #         num_rubrics_dict[int(rubric_sample['task_id'])] = len(rubric_sample['rubrics']['rubric_json'])

    ipdb.set_trace()
    pass




