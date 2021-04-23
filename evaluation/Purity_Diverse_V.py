import numpy as np
from sklearn import metrics

def get_Diverse_Score(label_true, label_pred):
    """
    calculate the Diverse Score(information entropy) of every true id
    :param label_true:
    :param label_pred:
    :return:
    """
    assert len(label_true) == len(label_pred)
    true_ids = np.unique(label_true)
    purity_score = dict()
    for t_id in true_ids:
        lt_pos = np.where(label_true == t_id)  # get current t_id positions in label_true
        n = len(lt_pos)  # n items in label_true with label==t_id
        pred_vals = label_pred[lt_pos]
        res_ids = np.unique(pred_vals)  # the unique ids in pred_vals
        _p_score = 0
        for r_id in res_ids:
            _p_score -= sum(res_ids == r_id) / n * np.log(sum(res_ids == r_id) / n)  # p*log(p)
        purity_score[t_id] = _p_score

    return purity_score


def get_Purity_Score(label_true, label_pred):
    """
    calculate the Purity Score(information entropy) of every result id
    :param label_true:
    :param label_pred:
    :return:
    """
    assert len(label_true) == len(label_pred)
    res_ids = np.unique(label_pred)
    diverse_score = dict()
    for r_id in res_ids:
        lp_pos = np.where(label_pred == r_id)  # get current r_id positions in label_pred
        m = len(lp_pos)  # m items in label_pred with label==r_id
        true_vals = label_true[lp_pos]
        true_ids = np.unique(true_vals)  # the unique ids in true_vals
        _d_score = 0
        for t_id in true_ids:
            _d_score -= sum(true_ids == t_id) / m * np.log(sum(true_ids == t_id) / m)  # p*log(p)
        diverse_score[r_id] = _d_score

    return diverse_score

def V_Measure(label_true, label_pred):
    h = metrics.homogeneity_score(label_true, label_pred)
    c = metrics.completeness_score(label_true, label_pred)
    v = 2 * h * c / (h + c)

    return h, c, v