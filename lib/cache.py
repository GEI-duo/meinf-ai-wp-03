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
        parquet_path = self.storage_dir / f"{filename}.parquet"
        pickle_path = self.storage_dir / f"{filename}.pkl"
        try:
            if parquet_path.exists():
                return pd.read_parquet(parquet_path)
            elif pickle_path.exists():
                return pd.read_pickle(pickle_path)
            else:
                raise FileNotFoundError(f"DataFrame file not found at: {parquet_path} or {pickle_path}")
        except Exception as e:
            logging.error(f"Failed to load DataFrame from parquet. Error: {e}")
            if pickle_path.exists():
                return pd.read_pickle(pickle_path)
            else:
                raise FileNotFoundError(f"DataFrame file not found at: {pickle_path}")

    def save(self, filename: str, df: pd.DataFrame) -> None:
        """Save a DataFrame to a file."""
        parquet_path = self.storage_dir / f"{filename}.parquet"
        pickle_path = self.storage_dir / f"{filename}.pkl"
        try:
            df.to_parquet(parquet_path)
            logging.info(f"DataFrame saved to {parquet_path}")
        except Exception as e:
            logging.error(f"Failed to save DataFrame to parquet. Error: {e}")
            df.to_pickle(pickle_path)
            logging.info(f"DataFrame saved to {pickle_path} as fallback")

def ensure_path_exists(path: Path) -> None:
    """Ensure that a path exists on the filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)

def cache(path, invalidate=False):
    ensure_path_exists(path)
    extension = os.path.splitext(path)[1]
    if extension == ".pkl":
        return cache_pickle(path, invalidate)
    elif extension == ".png":
        return cache_plot(path, invalidate)
    elif extension == ".joblib":
        return cache_joblib(path, invalidate)
    elif extension == ".npy":
        return cache_npy(path, invalidate)
    elif extension == ".parquet":
        return cache_parquet(path, invalidate)


def cache_plot(path, invalidate=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not os.path.exists(path) or invalidate:
                plot = func(*args, **kwargs)
                plt.savefig(path)
                return plot
            else:
                return Image(path)

        return wrapper

    return decorator


def cache_joblib(path, invalidate=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not os.path.exists(path) or invalidate:
                result = func(*args, **kwargs)
                dump(result, path)
                return result
            else:
                return load(path)

        return wrapper

    return decorator


def cache_pickle(path, invalidate=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not os.path.exists(path) or invalidate:
                print(
                    f"Cache not found or invalidated. Running function and saving to {path}..."
                )
                df = func(*args, **kwargs)
                df.to_pickle(path)
                print(f"Data cached successfully at {path}.")
                return df
            else:
                print(f"Loading data from cache: {path}")
                return pd.read_pickle(path)

        return wrapper

    return decorator


def cache_npy(path, invalidate=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not os.path.exists(path) or invalidate:
                print(
                    f"Cache not found or invalidated. Running function and saving to {path}..."
                )
                result = func(*args, **kwargs)
                np.save(path, result)
                print(f"Data cached successfully at {path}.")
                return result
            else:
                print(f"Loading data from cache: {path}")
                return np.load(path)

        return wrapper

    return decorator

def cache_parquet(path, invalidate=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not os.path.exists(path) or invalidate:
                print(
                    f"Cache not found or invalidated. Running function and saving to {path}..."
                )
                df = func(*args, **kwargs)
                df.to_parquet(path)
                print(f"Data cached successfully at {path}.")
                return df
            else:
                print(f"Loading data from cache: {path}")
                return pd.read_parquet(path)

        return wrapper

    return decorator