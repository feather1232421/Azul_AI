import pickle
import numpy as np

with open("greedy_scoring_dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

scores = [score for _, _, score in dataset]

print("num samples:", len(scores))
print("min score:", min(scores))
print("max score:", max(scores))
print("mean score:", np.mean(scores))
print("num zero:", sum(1 for s in scores if s == 0))
print("num positive:", sum(1 for s in scores if s > 0))
print("num negative:", sum(1 for s in scores if s < 0))