import pandas as pd
from sklearn.naive_bayes import MultinomialNB


DATA = "../^BVSP.csv"
TRAIN = "train.csv"
TEST = "test.csv"

def to_categorical(data):

    d = pd.read_csv(data)
    d = d.dropna()
    d = d.reset_index(drop=True)
    d = pd.DataFrame(d["Close"])

    ret = d.pct_change()
    ret = ret.drop(0).reset_index(drop=True)

    new_d = pd.DataFrame(
        index=range(len(d)),
        columns=["X1", "X2", "X3", "Label"]
    )

    for i, r in ret.iterrows():
        if i == 0:
            continue

        if i % 4 == 0:
            print(i-4, i)
            # x = new_d.loc[i-3:i]
        
    print(ret)

def split(data):
    size = 4

    with open(data, "r") as d:
        lines = d.readlines()

    clean = [l for l in lines if "null" not in l]
    train = []
    test = []

    i = 0
    for c in clean:
        if i < size:
            train += [c]
            i += 1
        else:
            test += [c]
            i = 0

    return train, test
        

def model():
    pass

def main():
    to_categorical(DATA)

if __name__ == "__main__":
    main()
