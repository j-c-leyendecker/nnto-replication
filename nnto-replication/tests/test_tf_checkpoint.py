"""The TensorFlow-free checkpoint reader returns exactly the saved weights."""
import os
import shutil

import numpy as np
import pytest

from nnto.tf_checkpoint import load_variables, read_index, crc32c, masked_crc32c


def test_crc32c_known_vector():
    # RFC 3720 test vector: 32 bytes of zeros -> 0x8A9136AA
    assert crc32c(bytes(32)) == 0x8A9136AA
    assert crc32c(b"123456789") == 0xE3069283


def test_shapes_and_values(upstream):
    v = load_variables(os.path.join(upstream, "NN_model_DBTO_1e+04", "variables"))
    s = "/.ATTRIBUTES/VARIABLE_VALUE"
    assert v["layer_with_weights-1/kernel" + s].shape == (1, 64)
    assert v["layer_with_weights-2/kernel" + s].shape == (64, 1)
    # input normalisation adapted on LHS samples of U(0,1): mean 1/2, var 1/12
    assert np.isclose(v["layer_with_weights-0/mean" + s][0], 0.5, atol=1e-6)
    assert np.isclose(v["layer_with_weights-0/variance" + s][0], 1 / 12, atol=1e-6)
    # output "normalisation" adapted on SIMP targets: mean 1/4, var 1/7 - 1/16
    assert np.isclose(v["layer_with_weights-3/mean" + s][0], 0.25, atol=1e-6)
    assert np.isclose(v["layer_with_weights-3/variance" + s][0], 1 / 7 - 1 / 16, atol=1e-6)
    # 5000 epochs x ceil(7000/32) = 1 095 000 Adam steps: no early stopping happened
    assert int(v["optimizer/_iterations" + s]) == 5000 * 219
    # statistics were adapted on train + validation samples (8500), not train only
    assert int(v["layer_with_weights-0/count" + s]) == 8500


def test_corruption_is_detected(upstream, tmp_path):
    src = os.path.join(upstream, "NN_model_HBTO_1e+04", "variables")
    dst = tmp_path / "variables"
    shutil.copytree(src, dst)
    data = dst / "variables.data-00000-of-00001"
    b = bytearray(data.read_bytes())
    b[40] ^= 0xFF                        # flip one byte inside the first kernel
    data.write_bytes(bytes(b))
    with pytest.raises(ValueError, match="CRC"):
        load_variables(str(dst))


def test_index_lists_expected_layers(upstream):
    idx = read_index(os.path.join(upstream, "NN_model_HBTO_1e+04", "variables", "variables.index"))
    kernels = sorted(k for k in idx if k.endswith("kernel/.ATTRIBUTES/VARIABLE_VALUE")
                     and k.startswith("layer_with_weights"))
    assert [idx[k]["shape"] for k in kernels] == [(2, 64), (64, 4)]
