import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():

    df = pd.read_csv(os.path.join("results", "fit_LearnerQ.csv"))
    # print(max(df["init_forget"]))

    plt.hist(df["init_forget"], log=True)
    plt.show()

    # plt.hist(df["rep_effect"])
    # plt.show()

    # plt.scatter(df["init_forget"], df["rep_effect"], alpha=0.01)
    # plt.show()


if __name__ == "__main__":
    main()