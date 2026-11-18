"""OPTIONAL cross-check with real TensorFlow/Keras (NOT executed in the
environment where this package was developed -- TensorFlow was unavailable).

It demonstrates audit finding A1 directly on the authors' saved model:
``Model.predict`` rounds float64 NumPy inputs to ``keras.backend.floatx()``
(float32 unless ``set_floatx('float64')`` is called, which the Run_* scripts
do not do), so the published central-difference derivative is 1.33 % too
large at theta = 0.5.

Expected output (from the NumPy re-implementation, tests/test_original_reproduction.py):

    floatx = float32 : CFD / exact  = 1.0132790
    floatx = float64 : CFD / exact  = 1.0000000

Run inside the authors' environment (Python 3.9.16, tensorflow 2.12.0):

    python tf_crosscheck_float32.py /path/to/NN_derivatives_TO_v3_final
"""
import sys

import numpy as np


def main(repo):
    import tensorflow as tf

    h = 1e-6
    x = np.full((4, 1), 0.5)
    for floatx in ("float32", "float64"):
        tf.keras.backend.set_floatx(floatx)
        model = tf.keras.models.load_model(repo + "/NN_model_DBTO_1e+04")
        cfd = (model.predict(x + h, verbose=0) - model.predict(x - h, verbose=0)) / (2 * h)
        xt = tf.Variable(x, dtype=tf.float64)
        with tf.GradientTape() as tape:
            y = xt
            for layer in model.layers:
                y = layer(y)
        exact = tape.batch_jacobian(y, xt).numpy()[:, 0, 0]
        print(f"floatx = {floatx} : CFD / exact = {cfd[0, 0] / exact[0]:.7f}")


if __name__ == "__main__":
    main(sys.argv[1])
