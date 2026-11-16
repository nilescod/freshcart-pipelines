
if "data_loader" not in globals():
    try:
        from mage_ai.data_preparation.decorators import data_loader
    except ImportError:
        def data_loader(function):
            return function

from sklearn.datasets import load_iris as sklearn_load_iris


@data_loader
def load_iris(*args, **kwargs):
    dataset = sklearn_load_iris(as_frame=True)
    df = dataset.frame.copy()

    print(f"Loaded {len(df)} Iris records")
    print(df.head())

    return df
