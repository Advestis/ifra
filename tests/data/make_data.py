import pandas as pd

n_nodes = 3
x = pd.read_csv("x.csv", header=None)
print("min: ", x.min())
print("max: ", x.max())
y = pd.read_csv("y.csv", header=None)

n_observations = len(y.index)

print("observations:", n_observations)

for i in range(n_nodes):
    subx = x.sample(frac=1 / (n_nodes - i))
    index = subx.index
    subx = subx.reset_index(drop=True)
    x = x.drop(index)

    suby = y.loc[index]
    suby = suby.reset_index(drop=True)

    subx.to_csv(f"node_{i}/x.csv")
    suby.to_csv(f"node_{i}/y.csv")
