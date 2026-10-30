"""Small variational quantum classifier (PennyLane simulator, default.qubit).

Enhancements over v1: mini-batch training, per-epoch loss history, optional
Platt calibration so the output is a real probability, and a score() helper.
"""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.linear_model import LogisticRegression


class QMLClassifier:
    def __init__(self, feature_idx, layers=3, epochs=15, batch_size=64, lr=0.1,
                 seed=42, calibrate=True):
        self.feature_idx = list(feature_idx)
        self.n = len(self.feature_idx)
        self.layers, self.epochs, self.batch_size = layers, epochs, batch_size
        self.lr, self.seed, self.calibrate = lr, seed, calibrate
        self.weights = None
        self.calibrator = None
        self.loss_history_ = []
        self._circuit = None

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_circuit"] = None          # qnodes cannot be pickled; rebuilt lazily
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
        return np.arctan(np.asarray(X, dtype=float)[:, self.feature_idx]) * 2.0

    def _raw(self, X):
        return np.asarray(self._get_circuit()(self._encode(X), self.weights))

    def fit(self, X, y):
        circuit = self._get_circuit()
        Xe = self._encode(X)
        y = np.asarray(y)
        target = np.where(y == 1, -1.0, 1.0)          # <Z> = -1  ->  class 1
        rng = np.random.default_rng(self.seed)
        w = pnp.array(rng.normal(0, 0.1, qml.StronglyEntanglingLayers.shape(self.layers, self.n)),
                      requires_grad=True)
        opt = qml.AdamOptimizer(self.lr)
        self.loss_history_ = []
        for _ in range(self.epochs):
            order = rng.permutation(len(Xe))
            losses = []
            for s in range(0, len(order), self.batch_size):
                b = order[s:s + self.batch_size]
                xb = pnp.array(Xe[b], requires_grad=False)
                tb = pnp.array(target[b], requires_grad=False)

                def cost(w, xb=xb, tb=tb):
                    return pnp.mean((circuit(xb, w) - tb) ** 2)
                w, loss = opt.step_and_cost(cost, w)
                losses.append(float(loss))
            self.loss_history_.append(float(np.mean(losses)))
        self.weights = np.array(w)
        if self.calibrate and len(np.unique(y)) == 2:
            raw = self._raw(X).reshape(-1, 1)
            self.calibrator = LogisticRegression().fit(-raw, y)   # Platt scaling
        return self

    def predict_proba(self, X):
        raw = self._raw(X)
        if self.calibrator is not None:
            return self.calibrator.predict_proba(-raw.reshape(-1, 1))
        p1 = np.clip((1 - raw) / 2, 0, 1)
        return np.stack([1 - p1, p1], axis=1)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))
