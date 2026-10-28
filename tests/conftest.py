import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def make_readings():
    """Factory for synthetic VM readings with the same schema as the challenge release."""

    def _make(n_vms: int = 20, n_ts: int = 134, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        timestamps = np.arange(n_ts) * 300
        level = rng.gamma(shape=1.5, scale=6.0, size=n_vms)
        avg = np.clip(level[:, None] + rng.normal(0.0, 1.5, size=(n_vms, n_ts)), 0.0, 100.0)
        frame = pd.DataFrame(
            {
                "timestamp": np.tile(timestamps, n_vms),
                "vm_id": np.repeat([f"vm{i}" for i in range(n_vms)], n_ts),
                "min_cpu": np.clip(avg - 1.0, 0.0, 100.0).ravel(),
                "max_cpu": np.clip(avg + 1.0, 0.0, 100.0).ravel(),
                "avg_cpu": avg.ravel(),
            }
        )
        # shuffle rows: the pipeline must not rely on the file being sorted
        return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return _make
