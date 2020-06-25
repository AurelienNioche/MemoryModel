import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "MemoryModel.settings")
django.setup()

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.special import expit

from data_interface.models import Data


from scipy.optimize import minimize, differential_evolution

EPS = np.finfo(np.float).eps
N_SEC_PER_DAY = 86400


class Learner:

    bounds = np.array([(0, 10**8), (0, 1)])
    init_guess = np.array([1.0, 0.00])
    param_labels = ("init_forget", "rep_effect")

    @classmethod
    def p(cls, param, n_rep, delta_last, outcome, *args, **kwargs):

        init_forget, rep_effect = param
        fr = init_forget * (1 - rep_effect) ** (n_rep - 2)

        p = np.exp(- fr * delta_last / N_SEC_PER_DAY)

        failure = np.invert(outcome)
        p[failure] = 1 - p[failure]
        p[p < 0] = 0
        return p

    @classmethod
    def objective(cls, param, kwargs):

        p = cls.p(param=param, **kwargs)
        ll = np.log(p + EPS)

        return -ll.sum()


class LearnerQ(Learner):

    bounds = np.array([(0, 10**6), ])
    init_guess = np.array([1.0, ])
    param_labels = ("init_forget", )

    @classmethod
    def p(cls, param, delta_last, outcome, deck, *args, **kwargs):

        init_forget = param
        fr = init_forget / deck

        # init_forget, rep_effect = param
        # fr = init_forget * (1 - rep_effect) ** (n_rep - 2)

        p = np.exp(- fr * delta_last / N_SEC_PER_DAY)

        failure = np.invert(outcome)
        p[failure] = 1 - p[failure]
        p[p < 0] = 0
        return p


class ActR(Learner):

    bounds = np.array([(0.0, 10**6),
                       (-10**6, 10**6),
                       (0.0001, 10**6),
                       (0.0, 10**6)])

    init_guess = np.array([0.5, 0., 10., 100.])
    param_labels = ("d", "tau", "s", "init_pe")

    @classmethod
    def p(cls, param, timestamp, delta_last, outcome, *args, **kwargs):

        d, tau, s, init_pe = param
        temp = s * np.square(2)

        n = len(timestamp)

        _pe = init_pe
        pe = np.zeros(n)

        for i in np.argsort(timestamp):
            with np.errstate(over='ignore'):
                _pe += (delta_last[i] / N_SEC_PER_DAY) ** (-d)
            pe[i] = _pe

        with np.errstate(divide='ignore'):
            a = np.log(pe)

        x = (- tau + a) / temp

        p = expit(x)
        failure = np.invert(outcome)
        p[failure] = 1 - p[failure]
        p[p > 1] = 1
        p[p < 0] = 0
        return p


class PowerLaw(Learner):
    bounds = np.array([(0.0, 10 ** 6),
                       (0.0, 10 ** 6),
                       (0.0, 10 ** 6)])

    init_guess = np.array([0.5, 1., 0.])
    param_labels = ("d", "a", "init_pe")

    @classmethod
    def p(cls, param, timestamp, delta_last, outcome, *args, **kwargs):
        d, a, init_pe = param

        n = len(timestamp)

        _pe = init_pe
        pe = np.zeros(n)

        for i in np.argsort(timestamp):
            with np.errstate(over='ignore'):
                _pe += a * (delta_last[i] / N_SEC_PER_DAY) ** (-d)
            pe[i] = _pe

        p = pe / (1 + pe)

        failure = np.invert(outcome)
        p[failure] = 1 - p[failure]
        p[p > 1] = 1
        p[p < 0] = 0
        return p


class Exponential(Learner):

    bounds = np.array([(0.0, 10 ** 6),
                       (0.0, 10 ** 6),
                       (0.0, 10 ** 6)])

    init_guess = np.array([0.5, 1., 0.])
    param_labels = ("d", "a", "init_pe")

    @classmethod
    def p(cls, param, timestamp, delta_last, outcome, *args, **kwargs):
        d, a, init_pe = param

        n = len(timestamp)

        _pe = init_pe
        pe = np.zeros(n)

        for i in np.argsort(timestamp):
            with np.errstate(over='ignore'):
                _pe += a * np.exp(- d * (delta_last[i] / N_SEC_PER_DAY))
            pe[i] = _pe

        p = pe / (1 + pe)

        failure = np.invert(outcome)
        p[failure] = 1 - p[failure]
        p[p > 1] = 1
        p[p < 0] = 0
        return p


def fit(class_model, args, method="SLSQP"):

    if method == "SLSQP":
        res = minimize(
            class_model.objective, x0=class_model.init_guess, args=args,
            bounds=class_model.bounds, method='SLSQP')
    elif method == "evolution":
        res = differential_evolution(
            func=class_model.objective, args=args,
            bounds=class_model.bounds)
    else:
        raise ValueError(f"Optimization method not recognized: '{method}'")

    n = len(args["timestamp"])
    lls = - res.fun
    # \mathrm{BIC} = k\ln(n) - 2\ln({\widehat{L}})
    k = len(class_model.param_labels)
    bic = k * np.log(n) - 2 * lls

    r = {}
    for k, v in zip(class_model.param_labels, res.x):
        r[k] = v
    r['n'] = n
    r['LLS'] = lls
    r['BIC'] = bic
    return r


def main():

    class_model = Exponential

    entries = Data.objects.exclude(n_rep=1)
    user_item_pair = \
        np.unique(entries.
                  values_list('user_item_pair_id', flat=True))
    black_list = \
        np.unique(entries.filter(delta_last=0)
                  .values_list('user_item_pair_id', flat=True))
    user_item_pair = list(user_item_pair)
    for b in black_list:
        user_item_pair.remove(b)

    results = []
    for uip in tqdm(user_item_pair):

        entries_uip = entries.filter(user_item_pair_id=uip)

        args = ({
            k: np.asarray(entries_uip.values_list(k, flat=True))
            for k in ("n_rep", "timestamp", "delta_last", "outcome", "deck")})

        r = fit(class_model=class_model, args=args)
        r['uip'] = uip
        results.append(r)

    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv(os.path.join("results", f"fit_{class_model.__name__}.csv"))


def sanity_check():
    pass

if __name__ == "__main__":
    main()
