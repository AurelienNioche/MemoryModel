import numpy as np

from scipy.special import expit
N_SEC_PER_DAY = 60*60*24


class ActR:

    bounds = np.array([(0.001, 1.0),
                       (-100., 10.),
                       (0.001, 5.),
                       (0.0, 10000)])

    init_guess = np.array([0.5, 0., 10., 100.])
    param_labels = ("d", "tau", "s", "init_pe")

    @classmethod
    def p(cls, param, timestamp, delta_last, outcome, *args, **kwargs):

        d, a, init_pe = param

        n = len(timestamp)

        _pe = init_pe
        pe = np.zeros(n)

        for i in np.argsort(timestamp):
            _pe += a * (delta_last[i]) ** (-d)
            pe[i] = _pe
        #
        # with np.errstate(divide='ignore'):
        #     a = np.log(pe)
        #
        # x = (- tau + a) / temp

        p = pe / (1 + pe)
        failure = np.invert(outcome)
        p[failure] = 1 - p[failure]
        p[p < 0] = 0
        return p


def main():

    print(ActR.p(param=(0.5, 100, 0), timestamp=np.arange(4), outcome=np.ones(4, dtype=bool),
                 delta_last=np.array([60, 60, 60, 60])))


if __name__ == "__main__":
    main()
