import numpy as np

from scipy.special import expit
N_SEC_PER_DAY = 60*60*24


class Learner:
    bounds = np.array([(0.0, 10 ** 6),
                       (0.0, 10 ** 6)])

    init_guess = np.array([0.5, 1.])
    param_labels = ("d", "a",)

    @classmethod
    def p(cls, param, timestamp, delta_last, outcome, *args, **kwargs):
        d, a = param

        n = len(timestamp)

        pe = np.zeros(n)

        sorted_idx = np.argsort(timestamp)
        timestamp = timestamp[sorted_idx]
        delta_last = delta_last[sorted_idx]
        outcome = outcome[sorted_idx]

        pe[0] = cls.f(delta_last[0], a, d)
        for i in range(1, n):
            delta = np.zeros(i)
            delta[:] = timestamp[i] - timestamp[:i]
            _pe = cls.f(delta, a, d)
            pe[i] = _pe.sum()

        p = pe / (1 + pe)

        failure = np.invert(outcome)
        p[failure] = 1 - p[failure]
        p[p > 1] = 1
        p[p < 0] = 0
        return p

    @classmethod
    def f(cls, delta, a, d):
        with np.errstate(over='ignore'):
            return a * (delta/N_SEC_PER_DAY) ** -d


def main():

    print(Learner.p(
        param=(0.5, 1,), timestamp=np.arange(4),
        outcome=np.ones(4, dtype=bool),
        delta_last=np.array([60, 60, 60, 60])))


if __name__ == "__main__":
    main()
