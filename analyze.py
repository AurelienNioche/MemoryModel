import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():

    models = "ActR2008", "ActR2005", "PowerLaw", "LearnerQ", "Learner"
    average_p = np.zeros(len(models))
    sum_lls = np.zeros(len(models))

    for i, model in enumerate(models):

        df = pd.read_csv(os.path.join("results", f"fit_{model}.csv"))
        # print(max(df["init_forget"]))

        col = [c for c in df.keys() if c not in ("LLS", "BIC", "n", "uip", "Unnamed: 0")]

        for c in col:

            plt.title(model)
            plt.hist(df[c], log=True, bins=20)
            plt.xlabel(c)
            plt.ylabel("freq")
            plt.show()

            # plt.scatter(df["init_forget"], df["rep_effect"], alpha=0.01)
            # plt.show()
        average_p[i] = np.exp(df["LLS"].sum() / df["n"].sum())
        sum_lls[i] = df["LLS"].sum()

    sorted_idx = np.argsort(-sum_lls)
    for i in sorted_idx:
        print("Model:", models[i])
        print("Average p:", average_p[i])
        print("Sum lls: ", sum_lls[i])
        print()


if __name__ == "__main__":
    main()
