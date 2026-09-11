import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# DBSCAN test results
# ==========================================

eps_values = [
    0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19,
    0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29,
    0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39,
    0.40
]

noise_points = [
    149037, 139761, 120781, 110103, 102037, 92514, 84779, 78531,
    67804, 56840, 49285, 44865, 41577, 38022, 35822, 33831,
    31456, 29639, 28008, 26799, 21438, 19470, 17245, 15540,
    14605, 13295, 12459, 11595, 10896, 10143, 9537
]

silhouette_scores = [
    0.2752, 0.3723, 0.3056, 0.4486, 0.4877, 0.4457,
    0.5836, 0.5599, 0.5420, 0.5250, 0.5149, 0.5031,
    0.4957, 0.4868, 0.4819, 0.4771, 0.4718, 0.4673,
    0.4636, 0.4599, 0.4430, 0.4422, 0.4412, 0.4379,
    0.4365, 0.4342, 0.4329, 0.4315, 0.4298, 0.4282,
    0.4267
]


# ==========================================
# Find best silhouette point
# ==========================================

best_index = np.argmax(silhouette_scores)

best_eps = eps_values[best_index]
best_noise = noise_points[best_index]
best_silhouette = silhouette_scores[best_index]


print("Best DBSCAN parameter:")
print(f"eps = {best_eps}")
print(f"Noise points = {best_noise}")
print(f"Silhouette score = {best_silhouette}")


# ==========================================
# Plot noise amount vs eps
# ==========================================

plt.figure(figsize=(10,6))

plt.plot(
    eps_values,
    noise_points,
    marker='o'
)


# Highlight highest silhouette point

plt.scatter(
    best_eps,
    best_noise,
    s=150,
    marker='*',
    label=f"Highest silhouette\n"
          f"eps={best_eps}, score={best_silhouette}"
)


plt.xlabel("eps")
plt.ylabel("Number of Noise Points")
plt.title("DBSCAN Noise Points vs eps (min_samples=1200)")

plt.grid(True)

plt.legend()

plt.show()
