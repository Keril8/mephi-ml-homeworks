from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Глобальный стиль графиков
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
BASE_DIR = Path(__file__).resolve().parent


class CustomLinearRegression:
    """
    Линейная регрессия на основе метода наименьших квадратов (Normal Equation).
    Реализована на чистом NumPy без высокоуровневых ML-библиотек.
    """

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.weights = None      # Вектор весов [w1, ..., wm]
        self.intercept = None    # Свободный член w0

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Обучение модели по формуле: w = (X^T * X)^(-1) * X^T * y
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        # Если передан одномерный массив признака (n,), преобразуем в (n, 1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]

        if self.fit_intercept:
            # Добавляем единичный столбец слева для свободного члена (bias)
            ones = np.ones((n_samples, 1))
            X_design = np.hstack([ones, X])
        else:
            X_design = X

        # Аналитическое решение нормального уравнения (OLS)
        # Используем np.linalg.pinv (псевдообратную матрицу) для вычислительной устойчивости
        # на случай мультиколлинеарности или вырожденности X^T * X
        w_optimal = np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y

        if self.fit_intercept:
            self.intercept = float(w_optimal[0, 0])
            self.weights = w_optimal[1:].flatten()
        else:
            self.intercept = 0.0
            self.weights = w_optimal.flatten()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Прогноз целевой переменной: y_pred = X @ w + intercept
        """
        if self.weights is None:
            raise ValueError("Модель ещё не обучена. Вызовите метод .fit() перед прогнозированием.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return (X @ self.weights + self.intercept).ravel()


# Модуль метрик качества регрессии

class RegressionMetrics:
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(RegressionMetrics.mse(y_true, y_pred)))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1.0 - (ss_res / ss_tot))

    @classmethod
    def print_report(cls, y_true: np.ndarray, y_pred: np.ndarray, title: str = "Regression Evaluation"):
        print(f"\n{'='*15} {title} {'='*15}")
        print(f"MSE:      {cls.mse(y_true, y_pred):.4f}")
        print(f"RMSE:     {cls.rmse(y_true, y_pred):.4f}")
        print(f"MAE:      {cls.mae(y_true, y_pred):.4f}")
        print(f"R² Score: {cls.r2_score(y_true, y_pred):.4f}")
        print(f"{'='*45}")


# Демонстрация и генерация синтетических данных

if __name__ == "__main__":
    np.random.seed(42)

    # 1. Генерация данных с шумом: y = 2.5 * x + 7.0 + noise
    TRUE_W1 = 2.5
    TRUE_W0 = 7.0
    N_SAMPLES = 150

    x = np.linspace(-10, 10, N_SAMPLES)
    noise = np.random.normal(loc=0.0, scale=4.0, size=N_SAMPLES)  # Гауссовский шум
    y = TRUE_W1 * x + TRUE_W0 + noise

    # 2. Обучение модели
    model = CustomLinearRegression(fit_intercept=True)
    model.fit(x, y)
    y_pred = model.predict(x)

    # 3. Вывод параметров и метрик
    print(f"Истинные параметры:     w0 = {TRUE_W0:.2f}, w1 = {TRUE_W1:.2f}")
    print(f"Найденные МНК веса:     w0 = {model.intercept:.4f}, w1 = {model.weights[0]:.4f}")

    RegressionMetrics.print_report(y, y_pred, title="Метрики на зашумлённых данных")

    # 4. Визуализация регрессионной прямой и остатков
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Левый график: точки + линия МНК
    axes[0].scatter(x, y, color="#2980b9", alpha=0.7, edgecolors="k", label="Выборка с шумом")
    axes[0].plot(x, y_pred, color="#e74c3c", linewidth=2.5, label=f"МНК линия: y = {model.weights[0]:.2f}x + {model.intercept:.2f}")
    axes[0].plot(x, TRUE_W1 * x + TRUE_W0, color="#27ae60", linestyle="--", linewidth=2, label="Истинная прямая")
    axes[0].set_title("Восстановление линейной зависимости (МНК)")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("y")
    axes[0].legend()

    # Правый график: распределение остатков (residuals)
    residuals = y - y_pred
    axes[1].scatter(y_pred, residuals, color="#8e44ad", alpha=0.7, edgecolors="k")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_title("Анализ остатков регрессии (Residuals vs Predicted)")
    axes[1].set_xlabel("Предсказанные значения (ŷ)")
    axes[1].set_ylabel("Остатки (y - ŷ)")

    plt.tight_layout()
    plt.savefig(BASE_DIR / "linear_regression_results.png", dpi=300)
    plt.show()