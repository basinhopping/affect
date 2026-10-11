"""Every component module, imported once so its registry entry exists.

Adding a dataset, feature or model means adding its module and one import line here.
"""


def load_all() -> None:
    from affect.data import iemocap  # noqa: F401  dataset "iemocap"
    from affect.features import logmel  # noqa: F401  feature "logmel"
    from affect.models import cnn  # noqa: F401  model "cnn"
    from affect.models import cnn_transformer  # noqa: F401  model "cnn_transformer"
    from affect.models import cnn_lif  # noqa: F401  model "cnn_lif" (CNN + LIF, hybrid)
    from affect.models import snn  # noqa: F401  model "snn" (CNN + LIF, fully spiking)
    from affect.classical import methods  # noqa: F401  classical methods "svm", "tpot"
