from collections import Counter
import pandas as pd
import numpy as np
from scipy import stats


def compute_probs(data, n=10): 
    h, e = np.histogram(data, n)
    p = h/data.shape[0]
    return e, p

def support_intersection(p, q): 
    sup_int = (
        list(
            filter(
                lambda x: (x[0]!=0) & (x[1]!=0), zip(p, q)
            )
        )
    )
    return sup_int


def get_probs(list_of_tuples): 
    p = np.array([p[0] for p in list_of_tuples])
    q = np.array([p[1] for p in list_of_tuples])
    return p, q


def kl_divergence(p, q): 
    return np.sum(p*np.log(p/q))


def js_divergence(p, q):
    m = (1./2.)*(p + q)
    return (1./2.)*kl_divergence(p, m) + (1./2.)*kl_divergence(q, m)


def compute_js_divergence(train_sample, test_sample, n_bins=10): 
    """
    Computes the JS Divergence using the support 
    intersection between two different samples
    """
    e, p = compute_probs(train_sample, n=n_bins)
    _, q = compute_probs(test_sample, n=e)
    
    list_of_tuples = support_intersection(p,q)
    p, q = get_probs(list_of_tuples)
    
    return js_divergence(p, q)


def rabbi_by_U(scores, reference_scores):
    U1, p = stats.mannwhitneyu(scores, reference_scores, method="asymptotic")
    effect_size = U1 / (len(scores) * len(reference_scores))
    return effect_size*2 - 1, p


def _order_ranks(ranks, j):
    ordered_ranks = np.empty(j.shape, dtype=ranks.dtype)
    np.put_along_axis(ordered_ranks, j, ranks, axis=-1)
    return ordered_ranks


def rabbi(scores, reference_scores):
    x = np.concatenate([scores, reference_scores])
    shape = x.shape
    j = np.argsort(x, axis=-1, kind="mergesort")
    ordinal_ranks = np.broadcast_to(np.arange(1, shape[-1]+1, dtype=int), shape)
    # Sort array
    y = np.take_along_axis(x, j, axis=-1)
    # Logical indices of unique elements
    i = np.concatenate([np.ones(shape[:-1] + (1,), dtype=np.bool_), y[..., :-1] != y[..., 1:]], axis=-1)

    # Integer indices of unique elements
    indices = np.arange(y.size)[i.ravel()]
    # Counts of unique elements
    counts = np.diff(indices, append=y.size)
    ranks = ordinal_ranks[i] + (counts - 1)/2
    ranks = np.repeat(ranks, counts).reshape(shape)
    ranks = _order_ranks(ranks, j)

    t = np.zeros(shape, dtype=float)
    t[i] = counts
    protected_ranks = ranks[:scores.shape[0]]
    reference_ranks = ranks[scores.shape[0]:]

    total = 0
    for r1 in protected_ranks:
        for r2 in reference_ranks:
            if r1 > r2:
                total +=1
            elif r1 < r2:
                total += -1
                
    rb = total/(scores.shape[0]*reference_scores.shape[0])
    return rb


def avg_score_gap(avg_scores_by_group, reference_group):
    reference_score = avg_scores_by_group[avg_scores_by_group.group == reference_group].score.item()
    avg_score_gap = avg_scores_by_group[avg_scores_by_group.group != reference_group].copy()
    avg_score_gap["avg_score_gap"] = avg_score_gap["score"] - reference_score
    return avg_score_gap[["group", "avg_score_gap"]]


def pairwise_avg_gap(pairwise_df, reference_group):
    groups = [g for g in pairwise_df["candidate_1.group"].unique() if g != reference_group]
    avg_gaps = []
    for group in groups:
        temp = pairwise_df[(pairwise_df["candidate_1.group"].isin([group, reference_group]))&(pairwise_df["candidate_2.group"].isin([group, reference_group]))]
        counts = Counter(temp.answer_group.tolist())
        avg_gaps.append((counts[group] - counts[reference_group])/temp.shape[0])
    return pd.DataFrame({"group": groups, "pairwise_avg_gap": avg_gaps})