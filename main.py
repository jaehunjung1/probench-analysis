import copy
from collections import defaultdict

import ipdb
import jsonlines
from tabulate import tabulate

from zhilin import get_predicted_score_per_task_id_e2e, get_std_at_k, get_pass_at_k


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
            total_task_id_to_scores[task_id].append(score)

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
    ipdb.set_trace()
    pass
