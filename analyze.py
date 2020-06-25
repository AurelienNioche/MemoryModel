import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():

    for model in "Exponential", :#"Learner", "LearnerQ":

        print(model)
        df = pd.read_csv(os.path.join("results", f"fit_{model}.csv"))
        # print(max(df["init_forget"]))

        col = [c for c in df.keys() if c not in ("LLS", "BIC", "n", "uip", "Unnamed: 0")]

        for c in col:

            plt.title(model)
            plt.hist(df[c], log=True)
            plt.xlabel(c)
            plt.ylabel("freq")
            plt.show()

            # plt.scatter(df["init_forget"], df["rep_effect"], alpha=0.01)
            # plt.show()

        print("Average p:", np.mean(np.exp(df["LLS"]/df["n"])))
        print("Sum lls: ", df["LLS"].sum())


if __name__ == "__main__":
    main()
