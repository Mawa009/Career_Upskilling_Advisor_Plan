
"""Member 3 inference interface for team integration."""

from pathlib import Path
from functools import lru_cache

from .career_paths import CareerTransitionModel
from .second_order import SecondOrderCareerModel


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"


@lru_cache(maxsize=1)
def get_first_order_model():
    path = MODEL_DIR / "transition_counts.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing model: {path}. Obtain the trained "
            "model artifact from Member 3."
        )
    return CareerTransitionModel.load(path)


@lru_cache(maxsize=1)
def get_second_order_model():
    path = MODEL_DIR / "second_order_counts.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing model: {path}. Obtain the trained "
            "model artifact from Member 3."
        )
    return SecondOrderCareerModel.load(path)


def predict_next_occupations(
    occupation_id,
    previous_occupation_id=None,
    top_k=5
):
    """
    Predict next occupations.

    If previous_occupation_id is provided, use the
    history-aware model; otherwise use first-order.

    Returns:
        List of dictionaries:
        occupation_id, label, probability
    """
    if previous_occupation_id is not None:
        model = get_second_order_model()
        return model.predict_next_occupations(
            current=occupation_id,
            previous=previous_occupation_id,
            top_k=top_k
        )

    model = get_first_order_model()
    return model.predict_next_occupations(
        occupation_id,
        top_k=top_k
    )
