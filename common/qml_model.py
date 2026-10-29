"""Small variational quantum classifier (PennyLane simulator, default.qubit)."""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class QMLClassifier:
    def __init__(self, feature_idx, layers=3, steps=100, lr=0.1, seed=42):
        self.feature_idx = list(feature_idx)
        self.n = len(self.feature_idx)
        self.layers, self.steps, self.lr, self.seed = layers, steps, lr, seed
        self.weights = None
        self._circuit = None

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_circuit"] = None
        return d

    def _get_circuit(self):
        if self._circuit is None:
            dev = qml.device("default.qubit", wires=self.n)

            @qml.qnode(dev, interface="autograd")
            def circuit(x, w):
                qml.AngleEmbedding(x, wires=range(self.n), rotation="Y")
                qml.StronglyEntanglingLayers(w, wires=range(self.n))
                return qml.expval(qml.PauliZ(0))
            self._circuit = circuit
        return self._circuit

    def _encode(self, X):
        return np.arctan(np.asarray(X)[:, self.feature_idx]) * 2.0

    def fit(self, X, y):
        circuit = self._get_circuit()
        Xe = pnp.array(self._encode(X), requires_grad=False)
        t = pnp.array(np.where(np.asarray(y) == 1, -1.0, 1.0), requires_grad=False)
        rng = np.random.default_rng(self.seed)
        w = pnp.array(rng.normal(0, 0.1, qml.StronglyEntanglingLayers.shape(self.layers, self.n)),
                      requires_grad=True)
        opt = qml.AdamOptimizer(self.lr)

        def cost(w):
            return pnp.mean((circuit(Xe, w) - t) ** 2)
        for _ in range(self.steps):
            w = opt.step(cost, w)
        self.weights = np.array(w)
        return self

    def predict_proba(self, X):
        out = np.asarray(self._get_circuit()(self._encode(X), self.weights))
        p1 = np.clip((1 - out) / 2, 0, 1)
        return np.stack([1 - p1, p1], axis=1)
