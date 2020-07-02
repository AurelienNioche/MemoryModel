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

    bounds = np.array([(0, 10**6), (0., 1.), ])
    init_guess = np.array([1.0, 0.00, ])
    param_labels = ("init_forget", "rep_effect")

    @classmethod
    def p(cls, param, n_rep, delta_last, outcome, *args, **kwargs):
        with np.errstate(divide='ignore', invalid='ignore'):
            init_forget, rep_effect = param
            fr = init_forget * (1 - rep_effect) ** (n_rep - 2)

            p = np.exp(- fr * delta_last)

            failure = np.invert(outcome)
            p[failure] = 1 - p[failure]
            p[p < 0] = 0
            p[p > 1] = 1
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
        with np.errstate(divide='ignore', invalid='ignore'):
            init_forget = param
            fr = init_forget / deck

            # init_forget, rep_effect = param
            # fr = init_forget * (1 - rep_effect) ** (n_rep - 2)

            p = np.exp(- fr * delta_last)

            failure = np.invert(outcome)
            p[failure] = 1 - p[failure]
            p[p < 0] = 0
            p[p > 1] = 1
        return p


class ActR2005(Learner):

    bounds = np.array([(-1, 1),
                       (0.00001, 1),
                       (0.0, 1.0),
                       (0.0000001, 100), ])

    init_guess = np.array([0.0, 1.0,  0.7, 1.0])
    param_labels = ("tau", "s", "a", "dt")

    @classmethod
    def p(cls, param, timestamp, delta_last, outcome, *args, **kwargs):
        with np.errstate(divide='ignore', invalid='ignore'):
            tau, s, a, dt = param

            n = len(timestamp)

            sorted_idx = np.argsort(timestamp)
            timestamp = timestamp[sorted_idx]
            delta_last = delta_last[sorted_idx]
            outcome = outcome[sorted_idx]

            e_m = np.zeros(n)
            delta0 = delta_last[0]
            timestamp0 = timestamp[0] - delta0
            timestamp = np.hstack((np.array([timestamp0, ]), timestamp))
            e_m[0] = delta0 ** -a
            for i in range(1, n):
                delta = timestamp[i + 1] - timestamp[:i + 1]
                e_m[i] = np.sum(np.power(dt * delta[:i + 1], a))

            x = (- tau + np.log(e_m)) / s

            p = expit(x)
            failure = np.invert(outcome)
            p[failure] = 1 - p[failure]
            p[p > 1] = 1
            p[p < 0] = 0
        return p


class ActR2008(Learner):

    bounds = np.array([(-1, 1),
                       (0.00001, 1),
                       (0.0, 1.0),
                       (0.0, 1.0),
                       (0.0000001, 10**4), ])

    init_guess = np.array([0.0, 1.0, 1.0, 0.7, 1.0])
    param_labels = ("tau", "s", "c", "a", "dt")

    @classmethod
    def p(cls, param, timestamp, delta_last, outcome, *args, **kwargs):
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            tau, s, c, a, dt = param

            n = len(timestamp)

            sorted_idx = np.argsort(timestamp)
            timestamp = timestamp[sorted_idx]
            delta_last = delta_last[sorted_idx]
            outcome = outcome[sorted_idx]

            d = np.full(n, a)
            e_m = np.zeros(n)
            delta0 = delta_last[0]
            timestamp0 = timestamp[0]-delta0
            timestamp = np.hstack((np.array([timestamp0, ]), timestamp))
            d[0] = a
            e_m[0] = delta0**-a
            for i in range(1, n):
                delta = timestamp[i+1] - timestamp[:i+1]
                d[i] = c * e_m[i - 1] + a
                e_m[i] = np.sum(np.power(dt*delta[:i+1], -d[:i+1]))

            x = (-tau + np.log(e_m)) / s

            p = expit(x)
            failure = np.invert(outcome)
            p[failure] = 1 - p[failure]
            p[p > 1] = 1
            p[p < 0] = 0
        return p


class PowerLaw(Learner):
    bounds = np.array([(0.0, 1.0),
                       (0.0, 1.0),
                       (0.0, 10 ** 4)])

    init_guess = np.array([0.5, 1.0, 1.0])
    param_labels = ("d", "a", "dt")

    @classmethod
    def p(cls, param, timestamp, delta_last, outcome, *args, **kwargs):
        with np.errstate(divide='ignore', invalid='ignore'):
            n = len(timestamp)

            d, a, dt = param

            sorted_idx = np.argsort(timestamp)
            timestamp = timestamp[sorted_idx]
            delta_last = delta_last[sorted_idx]
            outcome = outcome[sorted_idx]

            m = np.zeros(n)
            delta0 = delta_last[0]
            timestamp0 = timestamp[0]-delta0
            timestamp = np.hstack((np.array([timestamp0, ]), timestamp))

            m[0] = a * np.power(delta0*dt, -d)
            with np.errstate(divide='ignore', invalid='ignore'):
                for i in range(1, n):
                    delta = timestamp[i+1] - timestamp[:i+1]
                    m_ = a * np.power(delta*dt, -d)
                    m[i] = m_.sum()

            p = m / (1 + m)

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


def main(class_model):

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
        if entries_uip.count() < 20:
            continue

        args = ({
            k: np.asarray(entries_uip.values_list(k, flat=True))
            for k in ("n_rep", "timestamp", "delta_last", "outcome", "deck")})

        r = fit(class_model=class_model, args=args)
        r['uip'] = uip
        results.append(r)

    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv(os.path.join("results", f"fit_{class_model.__name__}.csv"))


if __name__ == "__main__":
    for cm in LearnerQ, Learner:
        main(class_model=cm)
