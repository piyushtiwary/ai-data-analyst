from states.root_state import RootState
import pandas as pd
import os


async def preparation_node(state: RootState):

    data_state = state["data"]

    file_path = data_state["file_path"]

    file_type = file_path.split(".")[-1].lower()

    raw_df = None

    # Load dataframe
    if file_type == "csv":
        raw_df = pd.read_csv(file_path)

    elif file_type == "json":
        raw_df = pd.read_json(file_path)

    elif file_type == "xlsx":
        raw_df = pd.read_excel(file_path)

    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    # Missing summary
    raw_missing_summary = raw_df.isnull().sum().to_dict()

    # Cleaning
    clean_df = raw_df.copy()

    clean_df.columns = [
        col.strip().lower().replace(" ", "_") for col in clean_df.columns
    ]

    clean_df = clean_df.drop_duplicates()

    clean_df = clean_df.dropna(how="all")

    for col in clean_df.select_dtypes(include="object").columns:
        clean_df[col] = clean_df[col].astype(str).str.strip()

    numeric_cols = clean_df.select_dtypes(include="number").columns

    for col in numeric_cols:
        clean_df[col] = clean_df[col].fillna(clean_df[col].median())

    categorical_cols = clean_df.select_dtypes(include="object").columns

    for col in categorical_cols:
        clean_df[col] = clean_df[col].fillna("Unknown")

    # Save cleaned dataset
    os.makedirs("outputs", exist_ok=True)

    clean_dataset_path = "outputs/clean_data.csv"

    clean_df.to_csv(clean_dataset_path, index=False)

    # Return ONLY updates
    return {
        "data": {
            **data_state,
            "file_type": file_type,
            "clean_dataset_path": clean_dataset_path,
            "raw_missing_summary": raw_missing_summary,
        }
    }
