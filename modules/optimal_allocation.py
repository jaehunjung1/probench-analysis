from typing import List

import numpy as np


def get_optimal_num_response_allocation(var_list: List[float], total_num_responses: int):
    M, K = len(var_list), total_num_responses
    min_num = 1  # minimum number of generations for all tasks
    extra_total = K - M * min_num  # 120
    INF = float('inf')

    # Initialize DP table for extras
    dp = np.full((M + 1, extra_total + 1), INF)
    dp[0, 0] = 0.0  # Base case: no tasks, no extras

    # Predecessor array to track extra allocations
    prev = np.full((M + 1, extra_total + 1), -1, dtype=int)

    # Fill DP table
    for m in range(1, M + 1):
        for k in range(0, extra_total + 1):  # k is total extras used so far
            for n_extra in range(0, k + 1):  # n_extra from 0 to k
                prev_k = k - n_extra
                if dp[m - 1, prev_k] != INF:
                    cost = dp[m - 1, prev_k] + var_list[m - 1] / (min_num + n_extra)
                    if cost < dp[m, k]:
                        dp[m, k] = cost
                        prev[m, k] = n_extra

    # Check if solution exists
    if dp[M, extra_total] == INF:
        print("No feasible solution found. Check constraints.")
        n_alloc = None
    else:
        # Backtrack to find extra allocations
        extra_alloc = [0] * M
        m, k = M, extra_total
        while m > 0:
            n_extra = prev[m, k]
            if n_extra == -1:
                print("Backtracking failed. Check DP logic.")
                break
            extra_alloc[m - 1] = n_extra
            k -= n_extra
            m -= 1

        n_alloc = [min_num + e for e in extra_alloc]
        # print("Optimal allocations:", n_alloc)
        # print("Total responses:", sum(n_alloc))
        # print("Optimal variance:", dp[M, extra_total] / (M ** 2))

    return n_alloc, dp[M, extra_total] / (M ** 2)