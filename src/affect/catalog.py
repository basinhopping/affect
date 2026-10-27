"""Every component module, imported once so its registry entry exists.

Adding a dataset, feature or model means adding its module and one import line here.
"""


def load_all() -> None:
    from affect.data import iemocap  # noqa: F401  dataset "iemocap"
    from affect.data import emofilm  # noqa: F401  dataset "emofilm" (a test corpus only; 2026-10-20)
    from affect.features import logmel  # noqa: F401  feature "logmel"
    from affect.features import frames as frame_features  # noqa: F401  feature "pretrained_frames" (2026-10-26)
    from affect.models import cnn  # noqa: F401  model "cnn"
    from affect.models import cnn_transformer  # noqa: F401  model "cnn_transformer"
    from affect.models import cnn_lif  # noqa: F401  model "cnn_lif" (CNN + LIF, hybrid)
    from affect.models import snn  # noqa: F401  model "snn" (CNN + LIF, fully spiking)
    from affect.models import cnn_rnn  # noqa: F401  model "cnn_rnn" (CNN + GRU or LSTM; 2026-10-24)
    from affect.models import frames as frame_heads  # noqa: F401  models "frames_*", heads on frames (2026-10-26)
    from affect.classical import methods  # noqa: F401  classical methods "svm", "tpot"
