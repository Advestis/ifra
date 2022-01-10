import pandas as pd

x = pd.read_csv("tests/data/x.csv", header=None)
y = pd.read_csv("tests/data/y.csv", header=None)

for i in range(4):
    subx = x.sample(frac=(i + 1)/4)
    index = subx.index
    subx = subx.reset_index(drop=True)
    x = x.drop(index)

    suby = y.loc[index]
    suby = suby.reset_index(drop=True)

    subx.to_csv(f"tests/data/node_{i}/x.csv")
    suby.to_csv(f"tests/data/node_{i}/y.csv")
