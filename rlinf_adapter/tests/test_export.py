import json

import numpy as np
import pytest
import tensorstore as ts

from b1k_rlinf.export import read_params


def test_ocdbt_restore_unwraps_nnx_and_uses_global_shape(tmp_path):
    params = tmp_path / "params"
    params.mkdir()
    name = "params.PaliGemma.img.embedding.kernel.value"
    expected = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    spec = {
        "driver": "zarr",
        "kvstore": {
            "driver": "ocdbt",
            "base": {"driver": "file", "path": str(params) + "/"},
            "path": name + "/",
        },
        "metadata": {"shape": list(expected.shape), "chunks": [1, 3, 4], "dtype": "<f4"},
        "create": True,
    }
    array = ts.open(spec).result()
    array.write(expected).result()
    keys = name.split(".")
    metadata = {
        "tree_metadata": {
            str(tuple(keys)): {
                "key_metadata": [{"key": key, "key_type": 2} for key in keys],
                # write_shape describes one training shard, not the whole tensor.
                "value_metadata": {
                    "value_type": "jax.Array",
                    "skip_deserialize": False,
                    "write_shape": [1, 3, 4],
                },
            }
        },
        "use_zarr3": False,
    }
    (params / "_METADATA").write_text(json.dumps(metadata))
    restored = read_params(tmp_path)
    np.testing.assert_array_equal(restored["PaliGemma"]["img"]["embedding"]["kernel"], expected)
    metadata["tree_metadata"][str(tuple(keys))]["key_metadata"][-2]["key"] = "missing"
    (params / "_METADATA").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Missing stored array"):
        read_params(tmp_path)
