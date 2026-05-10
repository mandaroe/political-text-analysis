from scipy.stats import wasserstein_distance
import numpy as np

labels = ["left", "leaning-left", "center", "leaning-right", "right"]
label_to_idx = {label: i for i, label in enumerate(labels)}

groups = [
    p_left,
    p_ll,
    p_center,
    p_lr,
    p_right
]

dist_matrix = np.zeros((len(labels), len(labels)))

for i, a in enumerate(groups):
    for j, b in enumerate(groups):
        dist_matrix[i, j] = wasserstein_distance(a, b)

print("\n" + "=" * 60)
print(f"Layer {layer_idx} - Wasserstein Distance Matrix")
print("=" * 60)

df = pd.DataFrame(dist_matrix, index=labels, columns=labels)
print(df)
