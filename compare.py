import argparse
from ast import literal_eval

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.contingency_tables import mcnemar


def get_success(file):
    """
    get successful instances of an algorithm
    :param file: the result file of the algorithm that you want to evaluate
    :return: a binary Numpy array where the position i is 1 if the i-th instance was successful
    """
    data = pd.read_csv(file)

    res = np.zeros(data.shape[0], dtype=np.bool_)

    for id, row in data.iterrows():
        user_id, item_id, topk, counterfactual, predicted_scores, replacement = row[:6]
        if not isinstance(counterfactual, str) or not isinstance(row['actual_scores_avg'], str):
            continue
        topk = literal_eval(topk)
        assert item_id == topk[0]
        actual_scores = literal_eval(row['actual_scores_avg'])

        replacement_rank = topk.index(replacement)
        if actual_scores[replacement_rank] > actual_scores[0]:
            res[id] = True
    return res


def get_size(file, mask):
    """
    get the size of explanations where the corresponding bit in mask is 1
    :param file: the result file of the algorithm that you want to evaluate
    :param mask: a bitmask where position i is set to 0 if instance i should not be considered
    :return: a Numpy array with size of all considered explanations
    """
    data = pd.read_csv(file)
    data = data[mask].reset_index(drop=True)

    res = np.zeros(data.shape[0], dtype=np.int32)
    for id, row in data.iterrows():
        user_id, item_id, topk, counterfactual, predicted_scores, replacement = row[:6]
        counterfactual = literal_eval(counterfactual)
        assert isinstance(counterfactual, set)
        res[id] = len(counterfactual)

    return res


def print_result(file):
    """
    print percentage and average size of counterfactual explanations
    :param file: the result file of the algorithm that you want to evaluate
    """
    res = get_success(file)
    print(f'{file}: {np.mean(res)}')

    algo_size = get_size(file, res)
    print(f'{file} size: {np.mean(algo_size)}')


def compare_algo(file, file2):
    """
    compare results of two algorithms. The function will print the following information:
    - The counterfactual percentage of each algorithm
    - A contingency table comparing two algorithms
    - The p-value of the McNemar's test comparing the counterfactual effect
    - The average counterfactual explanation size of each algorithm
    - The p-value of the t-test comparing these sizes.

    :param file: the result file of the first algorithm that you want to compare
    :param file2: the result file of the second algorithm that you want to compare
    """
    cont_table = np.zeros((2, 2))
    res = get_success(file)
    res2 = get_success(file2)
    for u, v in zip(res, res2):
        cont_table[1 - u, 1 - v] += 1
    print(f'{file}: {np.mean(res)}')
    print(f'{file2}: {np.mean(res2)}')
    print(cont_table)
    print(f'mcnemar: {mcnemar(cont_table).pvalue / 2}')

    both_ok = res & res2
    algo_size = get_size(file, both_ok)
    algo2_size = get_size(file2, both_ok)
    print(f'{file} size: {np.mean(algo_size)}')
    print(f'{file2} size: {np.mean(algo2_size)}')
    print(f't-test: {ttest_rel(algo_size, algo2_size)[1] / 2}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file")
    parser.add_argument("--file2")
    args = parser.parse_args()
    if args.file2 is None:
        print_result(args.file)
    else:
        compare_algo(args.file, args.file2)
