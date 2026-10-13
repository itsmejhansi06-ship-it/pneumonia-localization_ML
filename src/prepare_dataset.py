from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import IMAGE_DIR, LABEL_CSV, PROCESSED_ROOT
from .data_utils import load_labels, make_patient_table


def main():

    print("Loading labels...")

    # ---------------------------------------------------------
    # 1. Load original labels CSV
    # ---------------------------------------------------------
    df = load_labels(LABEL_CSV)

    patient_table = make_patient_table(df)

    print(f"Patients in CSV: {len(patient_table)}")

    # ---------------------------------------------------------
    # 2. Find available JPEG images
    # ---------------------------------------------------------
    available_images = {
        p.stem
        for p in Path(IMAGE_DIR).glob("*.jpg")
    }

    print(
        f"JPEG images found: {len(available_images)}"
    )

    # ---------------------------------------------------------
    # 3. Keep only patients having JPEG images
    # ---------------------------------------------------------
    patient_table = patient_table[
        patient_table["patientId"].isin(
            available_images
        )
    ].reset_index(drop=True)

    print(
        f"Patients with JPEG images: "
        f"{len(patient_table)}"
    )

    if len(patient_table) == 0:
        raise RuntimeError(
            "No matching JPEG images were found."
        )

    # ---------------------------------------------------------
    # 4. Overall class distribution
    # ---------------------------------------------------------
    print("\nOverall class distribution:")

    print(
        patient_table["Target"]
        .value_counts()
        .sort_index()
    )

    # ---------------------------------------------------------
    # 5. 70% Train / 30% Temporary
    # ---------------------------------------------------------
    train_df, temp_df = train_test_split(
        patient_table,
        test_size=0.30,
        random_state=42,
        stratify=patient_table["Target"]
    )

    # ---------------------------------------------------------
    # 6. 20% Validation / 10% Test
    #
    # temp = 30%
    #
    # test_size = 1/3
    #
    # Therefore:
    # validation = 20%
    # test       = 10%
    # ---------------------------------------------------------
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 / 3,
        random_state=42,
        stratify=temp_df["Target"]
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # ---------------------------------------------------------
    # 7. Display split sizes
    # ---------------------------------------------------------
    print("\nDataset split:")

    print(
        f"Train:       {len(train_df)}"
    )

    print(
        f"Validation:  {len(val_df)}"
    )

    print(
        f"Test:        {len(test_df)}"
    )

    # ---------------------------------------------------------
    # 8. Display class distributions
    # ---------------------------------------------------------
    print("\nTrain class distribution:")

    print(
        train_df["Target"]
        .value_counts()
        .sort_index()
    )

    print("\nValidation class distribution:")

    print(
        val_df["Target"]
        .value_counts()
        .sort_index()
    )

    print("\nTest class distribution:")

    print(
        test_df["Target"]
        .value_counts()
        .sort_index()
    )

    # ---------------------------------------------------------
    # 9. Create processed directory
    # ---------------------------------------------------------
    PROCESSED_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # 10. Save CSV files
    # ---------------------------------------------------------
    train_path = PROCESSED_ROOT / "train.csv"
    val_path = PROCESSED_ROOT / "val.csv"
    test_path = PROCESSED_ROOT / "test.csv"

    train_df.to_csv(
        train_path,
        index=False
    )

    val_df.to_csv(
        val_path,
        index=False
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    print("\nCSV files saved successfully:")

    print(train_path)
    print(val_path)
    print(test_path)


if __name__ == "__main__":
    main()