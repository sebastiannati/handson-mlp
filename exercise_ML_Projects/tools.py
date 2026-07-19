from pathlib import Path
import pandas as pd

def load_data_csv(path:str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.

    Parameters:
    path (str): The file path to the CSV file.

    Returns:
    pd.DataFrame: A DataFrame containing the loaded data.
    """
    path = Path(path)
    return pd.read_csv(path)
    