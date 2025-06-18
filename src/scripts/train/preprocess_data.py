"""
数据预处理脚本。

该脚本演示了如何使用 KuaiRandPreprocessor 类对 KuaiRand 数据集进行预处理。
"""

import os

<<<<<<< HEAD:src/scripts/train/preprocess_data.py
from src.features.kuairand_preprocess import (
    get_feature_columns,
    load_kuairand_data,
    preprocess_kuairand_features,
)


def main():
    """
    主函数，执行 KuaiRand 数据集的预处理流程。
    """
    # 定义数据集路径
    # 假设脚本在 RecJourney 目录下运行，KuaiRand-1K 在同级目录下
    data_root_path: str = "KuaiRand-1K"
    data_path: str = os.path.join("data", data_root_path, "data")

    # 检查数据路径是否存在
    if not os.path.exists(data_path):
        print(
            f"错误: 数据路径 '{data_path}' 不存在。请确保 KuaiRand-1K 数据集已下载并解压。"
        )
        print("请参考 KuaiRand-1K/README.md 下载数据。")
        return

    # 加载数据
    print("开始加载 KuaiRand-1K 数据...")
    try:
        log_df, user_df, video_basic_df, video_statistic_df = load_kuairand_data(
            data_path
        )
        print("数据加载完成。")
    except FileNotFoundError as e:
        print(f"加载数据文件时发生错误: {e}")
        print("请检查数据文件路径是否正确，并确保文件存在。")
        return
    except Exception as e:
        print(f"加载数据时发生未知错误: {e}")
        return
=======
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
>>>>>>> origin/refactor-ctr-models:main.py

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
