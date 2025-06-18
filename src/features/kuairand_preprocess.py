import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import MinMaxScaler


def load_kuairand_data(
    data_path: str = "KuaiRand-1K/data",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    加载 KuaiRand-1K 数据集。

    Args:
        data_path: 数据文件所在的目录路径。

    Returns:
        包含日志、用户特征、视频基本特征和视频统计特征的元组。
    """
    log_standard_1k_path = f"{data_path}/log_standard_4_08_to_4_21_1k.csv"
    log_standard_2_1k_path = f"{data_path}/log_standard_4_22_to_5_08_1k.csv"
    log_random_1k_path = f"{data_path}/log_random_4_22_to_5_08_1k.csv"
    user_features_1k_path = f"{data_path}/user_features_1k.csv"
    video_features_basic_1k_path = f"{data_path}/video_features_basic_1k.csv"
    video_features_statistic_1k_path = f"{data_path}/video_features_statistic_1k.csv"

    # 加载日志文件
    df_log_standard_1 = pd.read_csv(log_standard_1k_path)
    df_log_standard_2 = pd.read_csv(log_standard_2_1k_path)
    df_log_random = pd.read_csv(log_random_1k_path)
    df_log = pd.concat(
        [df_log_standard_1, df_log_standard_2, df_log_random], ignore_index=True
    )

    # 加载特征文件
    df_user_features = pd.read_csv(user_features_1k_path)
    df_video_basic_features = pd.read_csv(video_features_basic_1k_path)
    df_video_statistic_features = pd.read_csv(video_features_statistic_1k_path)

    return (
        df_log,
        df_user_features,
        df_video_basic_features,
        df_video_statistic_features,
    )


def _bin_numerical_feature(
    df: pd.DataFrame, column: str, num_bins: int = 10, strategy: str = "quantile"
) -> pd.DataFrame:
    """
    对数值特征进行分桶。

    Args:
        df: 输入 DataFrame。
        column: 需要分桶的列名。
        num_bins: 分桶的数量。
        strategy: 分桶策略，可以是 'quantile' (等频) 或 'width' (等宽)。

    Returns:
        分桶后的 DataFrame。
    """
    if column not in df.columns or df[column].isnull().all():
        print(f"警告: 列 '{column}' 不存在或全为 NaN，跳过分桶。")
        return df

    # 填充 NaN 值，以便分桶
    temp_series = df[column].fillna(df[column].median())

    if strategy == "quantile":
        # 等频分桶
        try:
            df[f"{column}_bin"] = pd.qcut(
                temp_series,
                q=num_bins,
                labels=False,
                duplicates="drop",
                precision=0,
            )
        except ValueError as e:
            print(f"警告: 对列 '{column}' 进行等频分桶时出错: {e}。尝试使用等宽分桶。")
            # 如果等频分桶失败（例如，数据中重复值过多），则回退到等宽分桶
            df[f"{column}_bin"] = pd.cut(
                temp_series, bins=num_bins, labels=False, precision=0
            )
    elif strategy == "width":
        # 等宽分桶
        df[f"{column}_bin"] = pd.cut(
            temp_series, bins=num_bins, labels=False, precision=0
        )
    else:
        raise ValueError("分桶策略必须是 'quantile' 或 'width'。")

    # 将分桶结果转换为整数类型，并将 NaN 填充为 -1（表示未分桶或原始值为 NaN）
    df[f"{column}_bin"] = df[f"{column}_bin"].fillna(-1).astype(int)
    return df


def preprocess_kuairand_features(
    df_log: pd.DataFrame,
    df_user_features: pd.DataFrame,
    df_video_basic_features: pd.DataFrame,
    df_video_statistic_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    对 KuaiRand 数据集进行特征预处理。

    Args:
        df_log: 日志 DataFrame。
        df_user_features: 用户特征 DataFrame。
        df_video_basic_features: 视频基本特征 DataFrame。
        df_video_statistic_features: 视频统计特征 DataFrame。

    Returns:
        预处理后的合并 DataFrame。
    """
    # 合并视频特征
    df_video_features = pd.merge(
        df_video_basic_features, df_video_statistic_features, on="video_id", how="left"
    )

    # 合并所有数据
    df_merged = pd.merge(df_log, df_user_features, on="user_id", how="left")
    df_merged = pd.merge(df_merged, df_video_features, on="video_id", how="left")

    # 处理时间特征
    df_merged["timestamp_s"] = df_merged["time_ms"] // 1000
    df_merged["datetime"] = pd.to_datetime(df_merged["timestamp_s"], unit="s")
    df_merged["dayofweek"] = df_merged["datetime"].dt.dayofweek
    df_merged["hour"] = df_merged["datetime"].dt.hour

    # 处理类别特征
    # 用户特征
    categorical_user_features = [
        "user_active_degree",
        "follow_user_num_range",
        "fans_user_num_range",
        "friend_user_num_range",
        "register_days_range",
    ]
    for col in categorical_user_features:
        df_merged[col] = df_merged[col].astype("category")
        df_merged[col] = df_merged[col].cat.codes  # 标签编码

    # 视频特征
    categorical_video_features = [
        "video_type",
        "upload_type",
        "music_type",
    ]
    for col in categorical_video_features:
        df_merged[col] = df_merged[col].astype("category")
        df_merged[col] = df_merged[col].cat.codes  # 标签编码

    # 处理多值类别特征 'tag'
    # 将tag字符串转换为列表，并进行独热编码（这里简化处理，实际可能需要更复杂的embedding）
    df_merged["tag"] = df_merged["tag"].apply(
        lambda x: [int(i) for i in str(x).split(",")] if pd.notna(x) else []
    )

    # 示例：处理布尔特征
    boolean_features = [
        "is_lowactive_period",
        "is_live_streamer",
        "is_video_author",
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_hate",
        "long_view",
        "is_profile_enter",
        "is_rand",
    ]
    for col in boolean_features:
        df_merged[col] = df_merged[col].astype(int)

    # 处理缺失值：对于数值特征，可以填充均值或中位数；对于类别特征，可以填充众数或特殊值
    # 这里简单填充0或-1，实际应用中需要更精细的策略
    numerical_features = df_merged.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    # 排除ID和时间戳，以及已经处理过的布尔特征
    exclude_cols = [
        "user_id",
        "video_id",
        "date",
        "hourmin",
        "time_ms",
        "timestamp_s",
        "visible_status",
        "server_width",
        "server_height",
        "music_id",
        "tab",
    ] + boolean_features
    numerical_features = [col for col in numerical_features if col not in exclude_cols]

    for col in numerical_features:
        if df_merged[col].isnull().any():
            df_merged[col] = df_merged[col].fillna(0)  # 示例填充0

    # 对选定的数值特征进行分桶
    features_to_bin = [
        "play_time_ms",
        "duration_ms",
        "profile_stay_time",
        "comment_stay_time",
        "follow_user_num",
        "fans_user_num",
        "friend_user_num",
        "register_days",
        "video_duration",
        "show_cnt",
        "play_cnt",
        "play_duration",
        "play_progress",
        "like_cnt",
        "comment_cnt",
        "share_cnt",
        "download_cnt",
        "collect_cnt",
    ]
    for col in features_to_bin:
        if col in df_merged.columns:
            df_merged = _bin_numerical_feature(df_merged, col, num_bins=10)

    # 数值特征归一化
    scaler = MinMaxScaler()
    # 归一化时，只对原始数值特征进行归一化，不包括新生成的分桶特征
    df_merged[numerical_features] = scaler.fit_transform(df_merged[numerical_features])

    # 对于类别特征，如果标签编码后有-1（表示NaN），可以进一步处理
    for col in categorical_user_features + categorical_video_features:
        if (df_merged[col] == -1).any():
            df_merged[col] = df_merged[col].replace(
                -1, 0
            )  # 将-1替换为0，或者其他合适的值

    # 删除原始的日期和时间列
    df_merged = df_merged.drop(columns=["date", "hourmin", "time_ms", "datetime"])

    return df_merged


def get_feature_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    获取特征列的分类。

    Args:
        df: 预处理后的 DataFrame。

    Returns:
        包含不同类型特征列名的字典。
    """
    feature_columns: Dict[str, List[str]] = {
        "user_id": ["user_id"],
        "video_id": ["video_id"],
        "label": ["is_click"],  # 假设is_click是标签
        "numerical_features": [],
        "categorical_features": [],
        "multivalued_features": ["tag"],
        "sequence_features": [],  # KuaiRand是顺序推荐数据集，但这里只做单点预处理，序列特征需要额外构建
    }

    # 识别数值特征
    numerical_cols = [
        "play_time_ms",
        "duration_ms",
        "profile_stay_time",
        "comment_stay_time",
        "follow_user_num",
        "fans_user_num",
        "friend_user_num",
        "register_days",
        "onehot_feat0",
        "onehot_feat1",
        "onehot_feat2",
        "onehot_feat3",
        "onehot_feat4",
        "onehot_feat5",
        "onehot_feat6",
        "onehot_feat7",
        "onehot_feat8",
        "onehot_feat9",
        "onehot_feat10",
        "onehot_feat11",
        "onehot_feat12",
        "onehot_feat13",
        "onehot_feat14",
        "onehot_feat15",
        "onehot_feat16",
        "onehot_feat17",
        "video_duration",
        "server_width",
        "server_height",
        "music_id",
        "tab",
        "counts",
        "show_cnt",
        "show_user_num",
        "play_cnt",
        "play_user_num",
        "play_duration",
        "complete_play_cnt",
        "complete_play_user_num",
        "valid_play_cnt",
        "valid_play_user_num",
        "long_time_play_cnt",
        "long_time_play_user_num",
        "short_time_play_cnt",
        "short_time_play_user_num",
        "play_progress",
        "comment_stay_duration",
        "like_cnt",
        "like_user_num",
        "click_like_cnt",
        "double_click_cnt",
        "cancel_like_cnt",
        "cancel_like_user_num",
        "comment_cnt",
        "comment_user_num",
        "direct_comment_cnt",
        "reply_comment_cnt",
        "delete_comment_cnt",
        "delete_comment_user_num",
        "comment_like_cnt",
        "comment_like_user_num",
        "follow_cnt",
        "follow_user_num",
        "cancel_follow_cnt",
        "cancel_follow_user_num",
        "share_cnt",
        "share_user_num",
        "download_cnt",
        "download_user_num",
        "report_cnt",
        "report_user_num",
        "reduce_similar_cnt",
        "reduce_similar_user_num",
        "collect_cnt",
        "collect_user_num",
        "cancel_collect_cnt",
        "cancel_collect_user_num",
        "direct_comment_user_num",
        "reply_comment_user_num",
        "share_all_cnt",
        "share_all_user_num",
        "outsite_share_all_cnt",
        "timestamp_s",
        "dayofweek",
        "hour",
    ]
    feature_columns["numerical_features"] = [
        col
        for col in numerical_cols
        if col in df.columns and col not in feature_columns["label"]
    ]

    # 识别类别特征 (标签编码后的)
    categorical_cols = [
        "user_active_degree",
        "is_lowactive_period",
        "is_live_streamer",
        "is_video_author",
        "follow_user_num_range",
        "fans_user_num_range",
        "friend_user_num_range",
        "register_days_range",
        "video_type",
        "upload_type",
        "music_type",
        "visible_status",
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_hate",
        "long_view",
        "is_profile_enter",
        "is_rand",
    ]
    # 添加分桶后的新类别特征
    binned_features = [col for col in df.columns if col.endswith("_bin")]
    categorical_cols.extend(binned_features)

    feature_columns["categorical_features"] = [
        col
        for col in categorical_cols
        if col in df.columns and col not in feature_columns["label"]
    ]

    return feature_columns


if __name__ == "__main__":
    print("开始加载 KuaiRand-1K 数据...")
    log_df, user_df, video_basic_df, video_statistic_df = load_kuairand_data()
    print("数据加载完成。")

    print("开始预处理特征...")
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
