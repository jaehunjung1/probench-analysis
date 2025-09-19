import copy
import random
from collections import defaultdict
from typing import List

import ipdb
import jsonlines
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tabulate import tabulate
import seaborn as sns

from zhilin import get_predicted_score_per_task_id_e2e


# def create_boxplot(task_score_list: List[List[int]], task_id_list, output_file="./images/boxplot.pdf"):
#     """
#     Create a box plot for 10 items, each with an empirical distribution of 16 floats,
#     and save it as a PDF.
#
#     Parameters:
#     - data: List of lists, where each inner list contains 16 floats for an item
#     - item_names: List of 10 strings, names for each item
#     - output_file: String, output PDF file name
#     """
#
#     # Convert data to DataFrame for seaborn
#     df = pd.DataFrame(task_score_list).T
#     df.columns = task_id_list
#
#     # Set seaborn style
#     sns.set_style("whitegrid")
#
#     # Create figure
#     plt.figure(figsize=(6, 6))
#
#     # Create box plot
#     sns.boxplot(data=df, orient="h")
#
#     # Customize plot
#     # plt.title("", fontsize=14, pad=15)
#     plt.xlabel("Score distribution", fontsize=10.5)
#     plt.ylabel("Task ID", fontsize=10.5)
#
#     # Adjust layout to prevent label cutoff
#     plt.tight_layout()
#
#     # Save as PDF
#     plt.savefig(output_file, format='pdf', bbox_inches='tight')
#     plt.close()


# def create_boxplot(task_score_list: List[List[int]], task_id_list, output_file="./images/boxplot.pdf"):
#     """
#     Create a box plot for 10 items, each with an empirical distribution of 16 floats,
#     and save it as a PDF.
#
#     Parameters:
#     - data: List of lists, where each inner list contains 16 floats for an item
#     - item_names: List of 10 strings, names for each item
#     - output_file: String, output PDF file name
#     """
#
#     # Convert data to DataFrame for seaborn
#     df = pd.DataFrame(task_score_list).T
#     df.columns = task_id_list
#
#     # Calculate means for each task and sort by mean
#     means = df.mean().sort_values()
#     sorted_columns = means.index
#     df = df[sorted_columns]
#
#     # Set seaborn style
#     sns.set_style("whitegrid")
#
#     # Create figure
#     plt.figure(figsize=(16, 4))
#
#     # Create box plot
#     sns.boxplot(data=df)
#
#     # Customize plot
#     # plt.title("", fontsize=14, pad=15)
#     plt.xlabel("Task ID", fontsize=12)
#     plt.ylabel("Score distribution", fontsize=12)
#
#     # Adjust layout to prevent label cutoff
#     plt.tight_layout()
#
#     # Save as PDF
#     plt.savefig(output_file, format='pdf', bbox_inches='tight')
#     plt.close()


def create_boxplot(task_score_list: List[List[int]], task_id_list, task_allocation_dict, output_file="./images/boxplot.pdf"):
    """
    Create a box plot for 10 items, each with an empirical distribution of 16 floats,
    sorted from low to high mean task, and save it as a PDF.

    Parameters:
    - task_score_list: List of lists, where each inner list contains 16 floats for an item
    - task_id_list: List of 10 strings, names for each item
    - output_file: String, output PDF file name
    """
    # Convert data to DataFrame for seaborn
    df = pd.DataFrame(task_score_list).T
    df.columns = [f"{task_id} ({task_allocation_dict[task_id]})" for task_id in task_id_list]

    # Melt the DataFrame to long format
    df_long = pd.melt(df, var_name='Task ID (# of responses in optimal allocation)', value_name='Score distribution')

    # Calculate means and sort from low to high
    mean_scores = df_long.groupby('Task ID (# of responses in optimal allocation)')['Score distribution'].mean().sort_values()
    order = mean_scores.index.tolist()

    # Set seaborn style
    sns.set_style("whitegrid")

    # Create figure
    plt.figure(figsize=(16, 4))

    # Create box plot with explicit order
    sns.boxplot(data=df_long, x='Task ID (# of responses in optimal allocation)', y='Score distribution', order=order,
                showfliers=True, color='lightblue')

    # Customize plot
    plt.xlabel("Task ID (# of responses in optimal allocation)", fontsize=12)
    plt.ylabel("Score distribution", fontsize=12)

    # xticks setting
    # plt.grid(axis='x')
    plt.gca().tick_params(axis='x', direction='out', bottom=True)
    plt.xticks(rotation=45, fontsize=10)
    # xtick_labels = plt.gca().get_xticklabels()
    # for label in xtick_labels:
    #     label.set_fontweight('bold')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save as PDF
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    filename = "data/standard_format_30_Aug_cleaned_w_filepath_filtered_infer_gemini-2.5-flash_reasoning_True_files_1_web_0_seed_16_model_nvdev_openai_gpt-oss-120b_reasoning_True.jsonl"
    K_list = [1, 2, 4, 8, 16]

    with jsonlines.open(filename) as f:
        data = list(f)
    data.sort(key=lambda x: x['idx'])

    total_performance = {
        'Physics PhD': [],
        'Chemistry PhD': [],
        'Investment Services': [],
        'Consulting': [],
        'Overall': [],
        'Extraction (recall)': [],
        'Reasoning': [],
        'Style': [],
        'completion_tokens': [],
    }
    total_task_id_to_scores = defaultdict(list)
    for batch_start_idx in range(0, len(data), 1162):
        batch = data[batch_start_idx:batch_start_idx + 1162]
        batch_performance, task_id_to_scores = get_predicted_score_per_task_id_e2e(condition="judge_rating", value="Yes", data=batch)
        for key in batch_performance:
            if key in total_performance:
                total_performance[key].append(batch_performance[key])

        for task_id, score in task_id_to_scores.items():
            total_task_id_to_scores[int(task_id)].append(score)

    # # Aggregation over K
    # for K in K_list:
    #     print("=" * 10 + f" K = {K}" + "=" * 10)
    #     for key, score_list in total_performance.items():
    #         print(f"{key}: {get_pass_at_k(score_list, K) :.3f} ({get_std_at_k(score_list, K) :.3f})")
    #
    # # Excluding min / max per individual column
    # for K in K_list[:-1]:
    #     print("=" * 10 + f" K = {K} (individual min / max exclusion) " + "=" * 10)
    #     for key, score_list in total_performance.items():
    #         min_score, max_score = min(score_list), max(score_list)
    #         reduced_score_list = [s for s in score_list if min_score < s < max_score]
    #         print(f"{key}: {get_pass_at_k(reduced_score_list, K) :.3f} ({get_std_at_k(reduced_score_list, K) :.3f})")
    #
    # # Excluding min / max based on overall
    # max_overall_batch_idx = total_performance["Overall"].index(max(total_performance["Overall"]))
    # min_overall_batch_idx = total_performance["Overall"].index(min(total_performance["Overall"]))
    # for K in K_list[:-1]:
    #     print("=" * 10 + f" K = {K} (overall min / max exclusion) " + "=" * 10)
    #     for key, score_list in total_performance.items():
    #         reduced_score_list = [s for idx, s in enumerate(score_list) if idx not in [max_overall_batch_idx, min_overall_batch_idx]]
        #         print(f"{key}: {get_pass_at_k(reduced_score_list, K) :.3f} ({get_std_at_k(reduced_score_list, K) :.3f})")

    quantile_results = []
    for task_id, scores in total_task_id_to_scores.items():
        scores = sorted(scores)
        p5, p25, p37_5, p50, p62_5, p75, p95 = scores[0], scores[3], scores[5], scores[7], scores[9], scores[11], scores[15]

        quantile_results.append({
            "task_id": task_id,
            "p5": p5, "p25": p25, "p37_5": p37_5, "p50": p50, "p62_5": p62_5, "p75": p75, "p95": p95,
        })
    print(tabulate(quantile_results, headers="keys", floatfmt=".3f"))

    task_id_list = sorted(list(total_task_id_to_scores.keys()))
    task_var_list = [float(np.var(total_task_id_to_scores[task_id])) for task_id in task_id_list]

    sampled_task_id_list = sorted(task_id_list, key=lambda x: np.mean(total_task_id_to_scores[x]))
    sampled_task_score_list = [total_task_id_to_scores[task_id] for task_id in sampled_task_id_list]
    task_allocation_dict = {
        2545: 4, 2561: 3, 2571: 4, 2592: 5, 2594: 4, 2598: 4, 2605: 4, 2626: 4, 2631: 5, 2638: 3, 2639: 4, 2641: 5,
        2649: 5, 2651: 5, 2664: 5, 2668: 4, 2677: 4, 2694: 5, 2698: 4, 2717: 4, 2721: 3, 2722: 3, 2728: 4, 2730: 4,
        2735: 3, 2743: 4, 2744: 4, 2751: 4, 2761: 4, 2786: 3, 2787: 4, 2791: 4, 2828: 4, 2829: 4, 2844: 4, 2894: 4,
        2900: 5, 2902: 4, 2915: 3, 2916: 3
    }

    create_boxplot(sampled_task_score_list, sampled_task_id_list, task_allocation_dict, output_file="./images/boxplot.pdf")