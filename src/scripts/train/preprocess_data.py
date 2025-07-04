"""
数据预处理脚本。

该脚本演示了如何使用 KuaiRandPreprocessor 类对 KuaiRand 数据集进行预处理。
"""

import os

import pandas as pd
from preprocess.data_processor import DataProcessor


def main():
    # read KuaiRand-1K/data/kuairand_merged_data.parquet
    data_path = "data/KuaiRand-1K/data/kuairand_merged_data.parquet"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # 为了测试，只加载少量数据
    df = pd.read_parquet(data_path).head(1000)
    print(f"Loaded data with shape: {df.shape}")
    print(df.head())
    print(f"Data columns: {df.columns.tolist()}")
    print("info:", df.info())

    print("\n--- Starting Data Preprocessing ---")
    config_dir = "config/dataset/KuaiRank-1K"
    processor = DataProcessor(config_dir)
    processed_df = processor.fit_transform(df)

    print("\n--- Preprocessing Finished ---")
    print(f"Processed data with shape: {processed_df.shape}")
    print(processed_df.head())
    print(f"Processed data columns: {processed_df.columns.tolist()}")
    print("Processed data info:", processed_df.info())

    print("\n--- Checking tag_processed column ---")
    if "tag_processed" in processed_df.columns:
        print(f"tag_processed dtype: {processed_df['tag_processed'].dtype}")
        print("tag_processed head:")
        print(processed_df["tag_processed"].head())
        print("tag_processed element types:")
        print(processed_df["tag_processed"].apply(type).value_counts())
    else:
        print("tag_processed column not found in processed data.")


if __name__ == "__main__":
    main()
