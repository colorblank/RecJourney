"""
主程序入口。

该脚本演示了如何使用 KuaiRandPreprocessor 类对 KuaiRand 数据集进行预处理。
"""

import os

import pandas as pd

from preprocess.kuairand_preprocess import (
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
    data_path: str = os.path.join(data_root_path, "data")

    # 检查数据路径是否存在
    if not os.path.exists(data_path):
        print(f"错误: 数据路径 '{data_path}' 不存在。请确保 KuaiRand-1K 数据集已下载并解压。")
        print("请参考 KuaiRand-1K/README.md 下载数据。")
        return

    # 加载数据
    print("开始加载 KuaiRand-1K 数据...")
    try:
        log_df, user_df, video_basic_df, video_statistic_df = load_kuairand_data(data_path)
        print("数据加载完成。")
    except FileNotFoundError as e:
        print(f"加载数据文件时发生错误: {e}")
        print("请检查数据文件路径是否正确，并确保文件存在。")
        return
    except Exception as e:
        print(f"加载数据时发生未知错误: {e}")
        return

    # 运行预处理
    print("开始预处理特征...")
    try:
        processed_df = preprocess_kuairand_features(
            log_df, user_df, video_basic_df, video_statistic_df
        )
        print("特征预处理完成。")
        print(f"预处理后的数据形状: {processed_df.shape}")
        print("预处理后的数据前5行:")
        print(processed_df.head())

        feature_cols_info = get_feature_columns(processed_df)
        print("\n特征列信息:")
        for k, v in feature_cols_info.items():
            print(f"{k}: {v}")

    except Exception as e:
        print(f"预处理过程中发生错误: {e}")


if __name__ == "__main__":
    main()
