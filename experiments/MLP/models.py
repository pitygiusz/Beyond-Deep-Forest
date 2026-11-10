"""Dataset-aware fully connected neural network classifiers."""

from sklearn.neural_network import MLPClassifier

from prepare_data import DATASETS, TEXT_DATASETS

IMAGE_DATASETS = {"cifar10", "mnist", "orl"}
SEQUENTIAL_DATASETS = {"semg", "gtzan"}

HIDDEN_LAYER_SIZES = {
    "tabular": (128, 128, 128, 64, 64, 64, 32, 32, 16, 16),
    "text": (256, 256, 256, 128, 128, 128, 64, 64, 32, 16),
    "image": (512, 512, 256, 256, 128, 128, 64, 64, 32, 16),
    "sequential": (256, 256, 256, 128, 128, 128, 64, 64, 32, 16),
}


def dataset_family(dataset):
    """Return the input family used to choose an MLP architecture."""
    normalized = str(dataset).strip().lower().replace("-", "_")
    if normalized in TEXT_DATASETS:
        return "text"
    if normalized in IMAGE_DATASETS:
        return "image"
    if normalized in SEQUENTIAL_DATASETS:
        return "sequential"
    if normalized not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {tuple(DATASETS)}")
    return "tabular"


def hidden_layer_sizes(dataset, n_layers=10):
    """Return a tapered, modality-specific architecture up to ten hidden layers."""
    if not 1 <= n_layers <= 10:
        raise ValueError("n_layers must be between 1 and 10")
    return HIDDEN_LAYER_SIZES[dataset_family(dataset)][:n_layers]


def make_mlp(
    dataset,
    *,
    n_layers=10,
    random_state=0,
    max_iter=300,
    alpha=1e-4,
    learning_rate_init=1e-3,
    early_stopping=True,
):
    """Build an MLP with up to ten hidden layers and inferred input/output widths."""
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes(dataset, n_layers),
        activation="relu",
        solver="adam",
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=early_stopping,
        random_state=random_state,
    )