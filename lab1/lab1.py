import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy import stats

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "cs2_all_tiers_games.csv")

# 1. Feature Engineering
t1_adr_cols = [f"team1_player{i}_adr" for i in range(1, 6)]
t2_adr_cols = [f"team2_player{i}_adr" for i in range(1, 6)]
t1_kills_cols = [f"team1_player{i}_kills" for i in range(1, 6)]
t2_kills_cols = [f"team2_player{i}_kills" for i in range(1, 6)]

df["team1_mean_adr"] = df[t1_adr_cols].mean(axis=1)
df["team2_mean_adr"] = df[t2_adr_cols].mean(axis=1)
df["team1_total_kills"] = df[t1_kills_cols].sum(axis=1)
df["team2_total_kills"] = df[t2_kills_cols].sum(axis=1)

df["total_rounds"] = df["score1_game"] + df["score2_game"]
df["adr_diff"] = df["team1_mean_adr"] - df["team2_mean_adr"]

# 2. Очистка от нулей и выбор топ-карт
top_maps = df["map_name"].value_counts().head(7).index

df_filtered = df[(df["map_name"].isin(top_maps)) & (df["total_rounds"] >= 13)].copy()

print("\nСтатистика после очистки:")
print(df_filtered[["total_rounds", "team1_mean_adr", "adr_diff"]].describe().T[["mean", "std", "50%", "min", "max"]])

# График 1: Популярность карт
plt.figure(figsize=(9, 4))
sns.countplot(
    data=df_filtered,
    x="map_name",
    order=top_maps,
    hue="map_name",
    palette="viridis",
    legend=False,
)
plt.title("Распределение сыгранных матчей по картам")
plt.xlabel("Карта")
plt.ylabel("Количество игр")
plt.tight_layout()
plt.savefig(BASE_DIR / "map_distribution.png", dpi=300)
plt.show()

# График 2: Длительность игр без нулей
plt.figure(figsize=(9, 4))
sns.histplot(df_filtered["total_rounds"], kde=True, discrete=True, color="#3498db")
plt.axvline(24, color="red", linestyle="--", label="Максимум основного времени (MR12: 24)")
plt.title("Распределение длительности игр (раундов на карте)")
plt.xlabel("Суммарное число раундов")
plt.ylabel("Частота")
plt.legend()
plt.tight_layout()
plt.savefig(BASE_DIR / "rounds_hist.png", dpi=300)
plt.show()

# График 3: Boxplot по картам
plt.figure(figsize=(10, 5))
sns.boxplot(
    data=df_filtered,
    x="map_name",
    y="total_rounds",
    hue="map_name",
    palette="Set2",
    legend=False,
)
plt.title("Разброс раундов на соревновательных картах")
plt.xlabel("Карта")
plt.ylabel("Всего раундов")
plt.tight_layout()
plt.savefig(BASE_DIR / "rounds_boxplot.png", dpi=300)
plt.show()

# График 4: # Разница ADR и победа
plt.figure(figsize=(7, 5))
sns.boxplot(
    data=df_filtered,
    x="team1_win",
    y="adr_diff",
    hue="team1_win",
    palette=["#e74c3c", "#2ecc71"],
    legend=False,
)
plt.xticks([0, 1], ["Поражение Team 1", "Победа Team 1"])
plt.title("Разница среднего командного ADR и исход карты")
plt.xlabel("Результат матча")
plt.ylabel("Разница по ADR")
plt.tight_layout()
plt.savefig(BASE_DIR / "adr_diff_boxplot.png", dpi=300)
plt.show()

# График 5: Корреляции
corr_cols = ["score1_game", "score2_game", "total_rounds", "team1_win", "team1_mean_adr", "team2_mean_adr", "adr_diff"]
plt.figure(figsize=(8, 6))
sns.heatmap(df_filtered[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Матрица корреляций ключевых признаков")
plt.tight_layout()
plt.savefig(BASE_DIR / "correlation_matrix.png", dpi=300)
plt.show()

print("\n" + "="*50)
print("СТАТИСТИЧЕСКАЯ ПРОВЕРКА ГИПОТЕЗЫ: ДЛИТЕЛЬНОСТЬ МАТЧЕЙ ПО КАРТАМ")
print("="*50)

# Группируем выборки раундов по каждой из 7 карт
map_samples = [
    group["total_rounds"].dropna().values
    for _, group in df_filtered.groupby("map_name")
]

# 1. Параметрический тест: One-way ANOVA
f_stat, p_val_f = stats.f_oneway(*map_samples)

# 2. Непараметрический аналог: Kruskal-Wallis (на случай скошенности распределения)
kw_stat, p_val_kw = stats.kruskal(*map_samples)

print(f"One-way ANOVA:      F = {f_stat:.4f}, p-value = {p_val_f:.4f}")
print(f"Kruskal-Wallis:     H = {kw_stat:.4f}, p-value = {p_val_kw:.4f}")