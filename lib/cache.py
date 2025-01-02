import os
from IPython.display import Image
import matplotlib.pyplot as plt
from joblib import dump, load
import pandas as pd
from pathlib import Path
import logging
import numpy as np


class DataFrameCache:
    def __init__(self, storage_dir: Path = Path("cache/dataframes")):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def load(self, filename: str) -> pd.DataFrame:
        """Load a DataFrame from a file."""
        full_path = self.storage_dir / f"{filename}.pkl"
        if not full_path.exists():
            raise FileNotFoundError(f"DataFrame file not found at: {full_path}")
        return pd.read_pickle(full_path)

    def save(self, filename: str, df: pd.DataFrame) -> None:
        """Save a DataFrame to a file."""
        full_path = self.storage_dir / f"{filename}.pkl"
        df.to_pickle(full_path)
        logging.info(f"DataFrame saved to {full_path}")


def cache(filename, invalidate=False):
    extension = os.path.splitext(filename)[1]
    if extension == ".pkl":
        return cache_pickle(filename, invalidate)
    elif extension == ".png":
        return cache_plot(filename, invalidate)
    elif extension == ".joblib":
        return cache_joblib(filename, invalidate)
    elif extension == ".npy":
        return cache_npy(filename, invalidate)


def cache_plot(filename, invalidate=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not os.path.exists(filename) or invalidate:
                plot = func(*args, **kwargs)
                plt.savefig(filename)
                return plot
            else:
                return Image(filename)

        return wrapper

    return decorator


def cache_joblib(filename, invalidate=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not os.path.exists(filename) or invalidate:
                result = func(*args, **kwargs)
                dump(result, filename)
                return result
            else:
                return load(filename)

        return wrapper

    return decorator


def cache_pickle(filename, invalidate=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not os.path.exists(filename) or invalidate:
                print(
                    f"Cache not found or invalidated. Running function and saving to {filename}..."
                )
                df = func(*args, **kwargs)
                df.to_pickle(filename)
                print(f"Data cached successfully at {filename}.")
                return df
            else:
                print(f"Loading data from cache: {filename}")
                return pd.read_pickle(filename)

        return wrapper

    return decorator


def cache_npy(filename, invalidate=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not os.path.exists(filename) or invalidate:
                print(
                    f"Cache not found or invalidated. Running function and saving to {filename}..."
                )
                result = func(*args, **kwargs)
                np.save(filename, result)
                print(f"Data cached successfully at {filename}.")
                return result
            else:
                print(f"Loading data from cache: {filename}")
                return np.load(filename)

        return wrapper

    return decorator
