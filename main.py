"""
主程序入口。

该脚本演示了如何使用 KuaiRandPreprocessor 类对 KuaiRand 数据集进行预处理。
"""

import os

import pandas as pd
from preprocess.data_processor import DataProcessor


def main():
    # read KuaiRand-1K/data/kuairand_merged_data.parquet
    data_path = "KuaiRand-1K/data/kuairand_merged_data.parquet"
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


if __name__ == "__main__":
    main()
