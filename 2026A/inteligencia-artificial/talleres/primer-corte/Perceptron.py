import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(
        self,
        n_inputs: int,
        learning_rate: float = 0.1,
        max_epochs: int = 100,
        random_state: int = 42,
    ):
        self.lr = learning_rate
        self.max_epochs = max_epochs
        rng = np.random.default_rng(random_state)
        self.weights: np.ndarray = rng.uniform(-0.5, 0.5, n_inputs)  # pesos iniciales aleatorios
        self.bias: float = 0.0
        self.error_history: list[int] = []

    def _step(self, z: np.ndarray) -> np.ndarray:
        # Función de activación escalón: 1 si z >= 0, 0 si no
        return (z >= 0).astype(int)

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.weights + self.bias  # combinación lineal: z = w·x + b
        return self._step(z)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.error_history = []

        for epoch in range(1, self.max_epochs + 1):
            epoch_errors = 0

            for xi, yi in zip(X, y):
                y_hat = self.predict(xi.reshape(1, -1))[0]
                error = int(yi) - int(y_hat)

                # Regla de Rosenblatt: actualiza solo si hay error
                self.weights += self.lr * error * xi
                self.bias    += self.lr * error

                if error != 0:
                    epoch_errors += 1

            self.error_history.append(epoch_errors)

            if epoch_errors == 0:  # convergencia: ningún error en la época
                print(f"  Convergió en la época {epoch}.")
                break
        else:
            print(f"  Máximo de épocas ({self.max_epochs}) alcanzado.")

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y)) * 100.0


def plot_decision_boundary(
    perceptron: Perceptron,
    X: np.ndarray,
    y: np.ndarray,
    title: str,
    ax: plt.Axes,
) -> None:
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Malla de puntos para colorear las regiones de cada clase
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25, cmap="bwr", levels=[-0.5, 0.5, 1.5])

    for clase, color, marker in [(0, "steelblue", "o"), (1, "tomato", "s")]:
        mask = y == clase
        ax.scatter(
            X[mask, 0], X[mask, 1],
            c=color, marker=marker, edgecolors="k",
            linewidths=0.6, s=70, label=f"Clase {clase}",
        )

    # Línea de decisión: w0*x1 + w1*x2 + b = 0  →  x2 = -(w0*x1 + b) / w1
    if abs(perceptron.weights[1]) > 1e-8:
        x_line = np.linspace(x_min, x_max, 200)
        y_line = -(perceptron.weights[0] * x_line + perceptron.bias) / perceptron.weights[1]
        ax.plot(x_line, y_line, "k--", linewidth=1.8, label="Frontera de decisión")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


def plot_errors(error_history: list[int], title: str, ax: plt.Axes) -> None:
    ax.bar(
        range(1, len(error_history) + 1),
        error_history,
        color="steelblue",
        edgecolor="k",
        linewidth=0.4,
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Época")
    ax.set_ylabel("Errores de clasificación")
    ax.grid(axis="y", alpha=0.3)


def demo_and():
    print("\n" + "=" * 55)
    print("DATASET 1 — Compuerta lógica AND")
    print("=" * 55)

    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y_and = np.array([0, 0, 0, 1])  # AND: solo (1,1) → 1

    p_and = Perceptron(n_inputs=2, learning_rate=0.1, max_epochs=100)
    p_and.train(X_and, y_and)

    preds = p_and.predict(X_and)
    df = pd.DataFrame({
        "x1"          : X_and[:, 0].astype(int),
        "x2"          : X_and[:, 1].astype(int),
        "AND esperado" : y_and,
        "Predicción"  : preds,
        "Correcto"    : ["Si" if p == r else "NO" for p, r in zip(preds, y_and)],
    })
    print(df.to_string(index=False))
    print(f"\nExactitud: {p_and.accuracy(X_and, y_and):.1f}%")

    return p_and, X_and, y_and


def demo_synthetic():
    print("\n" + "=" * 55)
    print("DATASET 2 — Clasificación 2D sintética (100 muestras)")
    print("=" * 55)

    rng = np.random.default_rng(0)
    n = 50

    # Dos nubes de puntos separadas: clase 0 en (1,1), clase 1 en (4,4)
    X0 = rng.normal(loc=[1.0, 1.0], scale=0.6, size=(n, 2))
    X1 = rng.normal(loc=[4.0, 4.0], scale=0.6, size=(n, 2))

    X_syn = np.vstack([X0, X1])
    y_syn = np.array([0] * n + [1] * n)

    idx = rng.permutation(len(y_syn))  # mezclar para no entrenar en orden de clase
    X_syn, y_syn = X_syn[idx], y_syn[idx]

    p_syn = Perceptron(n_inputs=2, learning_rate=0.1, max_epochs=100)
    p_syn.train(X_syn, y_syn)

    print(f"Exactitud: {p_syn.accuracy(X_syn, y_syn):.1f}%")

    return p_syn, X_syn, y_syn


def main():
    p_and, X_and, y_and = demo_and()
    p_syn, X_syn, y_syn = demo_synthetic()

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Perceptrón Simple — Resultados", fontsize=14, fontweight="bold")

    plot_decision_boundary(p_and, X_and, y_and, "AND — Frontera de decisión", axes[0, 0])
    plot_errors(p_and.error_history, "AND — Errores por época", axes[0, 1])

    plot_decision_boundary(p_syn, X_syn, y_syn, "Sintético 2D — Frontera de decisión", axes[1, 0])
    plot_errors(p_syn.error_history, "Sintético 2D — Errores por época", axes[1, 1])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
