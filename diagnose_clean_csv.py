import csv
import pandas as pd
import sys

CSV_PATH = 'final_annotations.csv'

def show_header_and_columns(path=CSV_PATH, n=30):
    print('--- raw header + first lines ---')
    try:
        with open(path, encoding='utf-8') as f:
            for i, line in enumerate(f, start=1):
                print(f"{i}: {line.rstrip()}")
                if i >= n:
                    break
    except Exception as e:
        print('Error reading raw file:', e)

    print('\n--- pandas read summary ---')
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[], na_filter=False)
        print('Columns:', df.columns.tolist())
        print('Dtypes:')
        print(df.dtypes)
        print('\nFirst 10 rows:')
        print(df.head(10).to_string())
        return df
    except Exception as e:
        print('Error reading with pandas:', e)
        return None


def find_missing_word_rows(df):
    if df is None:
        print('No dataframe to analyze')
        return
    mask_nan = df['word'].isna() if 'word' in df.columns else pd.Series([True]*len(df))
    mask_empty = df['word'].astype(str).str.strip() == '' if 'word' in df.columns else pd.Series([True]*len(df))
    mask_missing = mask_nan | mask_empty
    print('Total rows:', len(df))
    print('Rows missing word (NaN or empty):', mask_missing.sum())
    print(df[mask_missing].head(50).to_string())
    return mask_missing


def print_raw_lines_for_indices(indices, path=CSV_PATH):
    try:
        with open(path, encoding='utf-8') as f:
            for i, line in enumerate(f, start=1):
                if i in indices:
                    print(f"{i}: {line.rstrip()}")
    except Exception as e:
        print('Error reading file for indices:', e)


if __name__ == '__main__':
    df = show_header_and_columns()
    if df is None:
        sys.exit(1)
    mask = find_missing_word_rows(df)
    if mask is not None and mask.any():
        idxs = df[mask].index.tolist()[:20]
        # Convert df index to file line numbers (header=1 -> df index 0 maps to line 2)
        csv_lines = [i+2 for i in idxs]
        print('\nRaw CSV lines for first problematic entries (approx):', csv_lines)
        print_raw_lines_for_indices(csv_lines)
    else:
        print('No missing words detected')
