import pandas as pd
import numpy as np

direction_results = []

label_order = [
    "left",
    "leaning-left",
    "center",
    "leaning-right",
    "right"
]

for layer_idx in range(13):

    X = test_layers[layer_idx]

    left_mean = X[y_test_np == "left"].mean(axis=0)
    right_mean = X[y_test_np == "right"].mean(axis=0)

    ideology_direction = right_mean - left_mean

    ideology_direction = (
        ideology_direction /
        np.linalg.norm(ideology_direction)
    )

    projections = X @ ideology_direction

    layer_result = {
        "layer": layer_idx
    }

  # -----------------
  # Mean projection
  #------------------
  
    for label in label_order:

        idx = np.where(y_test_bias == label)[0]

        layer_result[label] = projections[idx].mean()

    direction_results.append(layer_result)

direction_df = pd.DataFrame(direction_results)

print("\n📈 Mean Directional Projections")
print(direction_df)
