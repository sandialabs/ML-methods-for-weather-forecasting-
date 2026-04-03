# src/io.py
from pathlib import Path
import pandas as pd

def read_csv_with_date(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
   
    # 1. Standardize 'date' to 'Date' if it exists
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'Date'})
   
    # 2. Convert to string if 'Date' is now in the columns
    if 'Date' in df.columns:
        df['Date'] = df['Date'].astype(str)
       
    return df

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def save_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=index)