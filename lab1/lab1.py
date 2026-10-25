import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Настройка визуального стиля графиков
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)

# -------------------------------------------------------------
# 1. ЗАГРУЗКА И ПЕРВИЧНЫЙ ОСМОТР
# -------------------------------------------------------------
df = pd.read_csv("lab1/cs2_all_tiers_games.csv")

print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} колонок")
print("\nПропущенные значения в основных полях:")
print(df[["tournament", "map_name", "score1_game", "score2_game", "team1_win"]].isna().sum())

# -------------------------------------------------------------
# 2. FEATURE ENGINEERING (Сборка командных агрегатов)
# Так как метрики разбиты по 5 игрокам, посчитаем средний ADR и Kills команды
# -------------------------------------------------------------
t1_adr_cols = [f"team1_player{i}_adr" for i in range(1, 6)]
t2_adr_cols = [f"team2_player{i}_adr" for i in range(1, 6)]

t1_kills_cols = [f"team1_player{i}_kills" for i in range(1, 6)]
t2_kills_cols = [f"team2_player{i}_kills" for i in range(1, 6)]

# Средний урон и суммарные убийства команд на карте
df["team1_mean_adr"] = df[t1_adr_cols].mean(axis=1)
df["team2_mean_adr"] = df[t2_adr_cols].mean(axis=1)
df["team1_total_kills"] = df[t1_kills_cols].sum(axis=1)
df["team2_total_kills"] = df[t2_kills_cols].sum(axis=1)

# Общее число раундов на карте
df["total_rounds"] = df["score1_game"] + df["score2_game"]

# Разница по среднему ADR между командами
df["adr_diff"] = df["team1_mean_adr"] - df["team2_mean_adr"]

# Фильтрация редких карт (оставим активный соревновательный пул)
top_maps = df["map_name"].value_counts().head(7).index
df_filtered = df[df["map_name"].isin(top_maps)].copy()

# -------------------------------------------------------------
# 3. СТАТИСТИЧЕСКИЙ АНАЛИЗ
# -------------------------------------------------------------
print("\nОписательная статистика ключевых числовых метрик:")
stats_cols = ["score1_game", "score2_game", "total_rounds", "team1_mean_adr", "adr_diff"]
print(df_filtered[stats_cols].describe().T[["mean", "std", "min", "50%", "max"]])

# -------------------------------------------------------------
# 4. ВИЗУАЛИЗАЦИЯ (EDA)
# -------------------------------------------------------------

# График 1: Распределение популярности соревновательных карт
plt.figure(figsize=(9, 4))
sns.countplot(data=df_filtered, x="map_name", order=top_maps, palette="viridis")
plt.title("Распределение сыгранных матчей по картам")
plt.xlabel("Карта")
plt.ylabel("Количество игр")
plt.tight_layout()
plt.show()

# График 2: Гистограмма и KDE распределения общего числа раундов
plt.figure(figsize=(9, 4))
sns.histplot(df_filtered["total_rounds"], kde=True, discrete=True, color="#3498db")
plt.axvline(24, color="red", linestyle="--", label="Стандартный максимум (MR12: 13+11=24)")
plt.title("Распределение длительности игр (количество раундов на карте)")
plt.xlabel("Суммарное число раундов")
plt.ylabel("Частота")
plt.legend()
plt.tight_layout()
plt.show()

# График 3: Boxplot длительности матчей в разрезе карт (поиск аномалий/овертаймов)
plt.figure(figsize=(10, 5))
sns.boxplot(data=df_filtered, x="map_name", y="total_rounds", palette="Set2")
plt.title("Разброс раундов на разных соревновательных картах")
plt.xlabel("Карта")
plt.ylabel("Всего раундов")
plt.tight_layout()
plt.show()

# График 4: Влияние преимущества по ADR на победу (Boxplot разницы ADR)
plt.figure(figsize=(7, 5))
sns.boxplot(data=df_filtered, x="team1_win", y="adr_diff", palette=["#e74c3c", "#2ecc71"])
plt.xticks([0, 1], ["Поражение Team 1", "Победа Team 1"])
plt.title("Разница среднего командного ADR (Team 1 - Team 2) и исход карты")
plt.xlabel("Результат матча для Team 1")
plt.ylabel("Разница по ADR")
plt.tight_layout()
plt.show()

# График 5: Корреляционная матрица (Heatmap)
corr_cols = [
    "score1_game", "score2_game", "total_rounds", 
    "team1_win", "team1_mean_adr", "team2_mean_adr", "adr_diff"
]
plt.figure(figsize=(8, 6))
sns.heatmap(df_filtered[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Матрица корреляций ключевых признаков матча")
plt.tight_layout()
plt.show()