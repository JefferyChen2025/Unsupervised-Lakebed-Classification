import matplotlib.pyplot as plt

# ==========================================
# DBSCAN eps and silhouette score data
# ==========================================

eps_values = [
    0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,
    0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23,
    0.24, 0.25, 0.26, 0.27, 0.28, 0.29,
    0.30, 0.31, 0.32, 0.33, 0.34, 0.35,
    0.36, 0.37, 0.38, 0.39, 0.40
]

silhouette_scores = [
    0.2752, 0.3723, 0.3056, 0.4486, 0.4877,
    0.4457, 0.5836, 0.5599, 0.5420, 0.5250,
    0.5149, 0.5031, 0.4957, 0.4868,
    0.4819, 0.4771, 0.4718, 0.4673,
    0.4636, 0.4599, 0.4430, 0.4422,
    0.4412, 0.4379, 0.4365, 0.4342,
    0.4329, 0.4315, 0.4298, 0.4282,
    0.4267
]

# ==========================================
# Find best silhouette score
# ==========================================

best_index = silhouette_scores.index(max(silhouette_scores))

best_eps = eps_values[best_index]
best_score = silhouette_scores[best_index]


# ==========================================
# Plot
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    eps_values,
    silhouette_scores,
    marker='o',
    linewidth=2
)

# Highlight maximum value
plt.scatter(
    best_eps,
    best_score,
    s=120
)

plt.annotate(
    f"Best eps={best_eps}\nSilhouette={best_score}",
    xy=(best_eps, best_score),
    xytext=(best_eps+0.03, best_score-0.05),
    arrowprops=dict(arrowstyle="->")
)


# Labels and title
plt.xlabel("DBSCAN eps value", fontsize=12)
plt.ylabel("Silhouette Score", fontsize=12)

plt.title(
    "DBSCAN Parameter Selection: eps vs Silhouette Score",
    fontsize=14
)

plt.grid(True)

plt.ylim(0, 0.7)

plt.show()
