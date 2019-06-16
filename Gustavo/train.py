import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


DATA = "../^BVSP.csv"

def to_categorical(data):

    # Clean up, select only Close column
    numeric_d = pd.read_csv(data)
    numeric_d = numeric_d.dropna()
    numeric_d = numeric_d.reset_index(drop=True)
    numeric_d = pd.DataFrame(numeric_d["Close"])

    # Calculate return
    pct_return = numeric_d.pct_change()
    pct_return = pct_return.drop(0).reset_index(drop=True)

    # Init new dataframe that holds categorical data
    categorical_d = pd.DataFrame(
        index=range(len(pct_return)),
        columns=["X1", "X2", "X3", "Label"]
    )

    end_idx = 0
    start_idx = 0
    i = 0
    while end_idx < len(pct_return):
        i += 1
        end_idx = start_idx + 4

        cur_df = pct_return.iloc[start_idx:end_idx].copy().reset_index(drop=True)

        if len(cur_df) == 4:
            x1 = cur_df.iloc[0]["Close"]
            x2 = cur_df.iloc[1]["Close"]
            x3 = cur_df.iloc[2]["Close"]
            label = cur_df.iloc[3]["Close"]

            categorical_d.iloc[i] = [x1, x2, x3, label]

        start_idx = end_idx

    categorical_d = categorical_d.dropna().reset_index(drop=True)
    c = pd.cut(
        categorical_d.stack(),
        [-np.inf, -0.01, -0.005, 0, 0.005, 0.01, np.inf],
        labels=[0, 1, 2, 3, 4, 5]
    )
    categorical_d = categorical_d.join(c.unstack().add_suffix('_cat'))
    categorical_d = categorical_d.drop(columns=["X1", "X2", "X3", "Label"])

    return categorical_d        

def main():
    cat = to_categorical(DATA)
    print(cat.info())
    clf = MultinomialNB()
    print(cat.iloc[:, 3])
    clf.fit(cat.iloc[:, :3], cat.iloc[:, 3])
    joblib.dump(clf, 'nb_clf.pickle')

if __name__ == "__main__":
    main()
