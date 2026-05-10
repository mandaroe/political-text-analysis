import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# ----------------------
# Ideological Direction
# ----------------------

fig, ax = plt.subplots(figsize=(10,6))
for label in label_order:
    ax.plot(
        direction_df["layer"],
        direction_df[label],
        marker="o",
        label=label
    )
  
ax.set_xlabel("BERT Layer")
ax.set_ylabel("Mean Projection onto Ideology Direction")
ax.set_title("Ideological Direction Across Layers")
ax.legend(title="Ideology")
ax.grid(alpha=0.3)

plt.show()

# -------------------
# Geometric Structure
# -------------------

v = ideology_direction / np.linalg.norm(ideological_direction)
y_labels_numeric = np.array([bias_label_map[label] for label in y_test_bias])

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

X_proj = X - (X @ v[:, None]) * v

pca = PCA(n_components=2)
orthogonal = pca.fit_transform(X_proj)

ideology_axis = X @ v

plt.figure(figsize=(7,6))

plt.scatter(
    ideology_axis,
    orthogonal[:,0],
    c=y_labels_numeric,
    cmap="coolwarm",
    alpha=0.6
)

plt.xlabel("Ideology Direction")
plt.ylabel("Orthogonal Variation")
plt.title("Geometric Structure of Ideological Space")

plt.show()

# ------------------
# Wassertein Heat Map
# ------------------

plt.figure(figsize=(7,6))

sns.heatmap(
    dist_matrix,
    xticklabels=labels,
    yticklabels=labels,
    annot=True,
    fmt=".2f",
    square=True
)

plt.title(f"Wasserstein Distance Matrix (Layer {layer_idx})")
plt.tight_layout()

plt.show()

# -----------------------------
# Distribution Overlap with KDE
# -----------------------------

plt.figure(figsize=(8,5))

sns.kdeplot(p_left, label="left")
sns.kdeplot(p_ll, label="leaning-left")
sns.kdeplot(p_center, label="center")
sns.kdeplot(p_lr, label="leaning-right")
sns.kdeplot(p_right, label="right")

plt.title("Ideology Projections with Distribution Overlap")
plt.xlabel("Projection onto Ideology Direction")
plt.ylabel("Density")
plt.legend()

plt.show()

os.makedirs("figures", exist_ok=True)
plt.savefig("figures/ideology_direction.png", dpi=300, bbox_inches="tight")
