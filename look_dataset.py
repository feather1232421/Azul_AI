import pickle

with open("MCTS_nn_dataset_pi.pkl", "rb") as f:
    dataset = pickle.load(f)

obs, pi, z, mask = dataset[10]
print(obs, pi, z)
print(len(dataset))
print(mask)
print(obs.shape, obs.dtype)
print(pi.shape, pi.dtype)
print("pi sum =", pi.sum())
print("pi argmax = ", pi.argmax())
print("pi max = ", pi.max())
print("pi nonzero =", (pi > 0).sum())
print("z =", z)