"""Baseline = the original repository's code, preserved for comparison.

``original``   executes the *unmodified* functions of the published scripts
               (extracted from the source files with ``ast``; nothing is
               copied or edited) against a NumPy stand-in for the handful of
               TensorFlow/Keras calls they make.
``keras_shim`` is that stand-in.  It reproduces the numerics that matter for
               the audit, in particular ``Model.predict`` casting NumPy inputs
               to ``keras.backend.floatx()`` (float32 unless changed).
"""
