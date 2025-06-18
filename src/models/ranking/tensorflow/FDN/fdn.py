import torch
import torch.nn as nn


class FDN(nn.Module):
    """
    实现因子分解网络 (Factorization Decomposition Network) 的 PyTorch 模型。

    参数:
        vocab_sizes (dict[str, int]): 特征词汇表大小的字典，键为特征 ID，值为词汇表大小。
        embedding_size (int): 嵌入层的维度大小。
        task_specific_units (list[int]): 任务特定专家网络的隐藏层单元数列表。
        shared_units (list[int]): 共享专家网络的隐藏层单元数列表。
        task_num (int): 任务数量。
    """

    def __init__(
        self,
        vocab_sizes: dict[str, int],
        embedding_size: int,
        task_specific_units: list[int],
        shared_units: list[int],
        task_num: int,
    ):
        """
        初始化 FDN 模型。

        参数:
            vocab_sizes (dict[str, int]): 特征词汇表大小的字典。
            embedding_size (int): 嵌入层的维度大小。
            task_specific_units (list[int]): 任务特定专家网络的隐藏层单元数列表。
            shared_units (list[int]): 共享专家网络的隐藏层单元数列表。
            task_num (int): 任务数量。
        """
        super(FDN, self).__init__()
        self.embedding_size = embedding_size
        self.task_num = task_num

        # 创建嵌入层的 ModuleDict
        self.embeddings = nn.ModuleDict(
            {
                f"feat_{feat_id}": nn.Embedding(vocab_size, embedding_size)
                for feat_id, vocab_size in vocab_sizes.items()
            }
        )

        # 创建任务特定专家网络的 ModuleList
        self.task_specific_experts = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(
                                embedding_size * len(vocab_sizes),
                                task_specific_units[0],
                            ),
                            nn.ELU(),
                            nn.Linear(task_specific_units[0], task_specific_units[1]),
                            nn.ELU(),
                        )
                        for _ in range(task_num)
                    ]
                )
                for _ in range(task_num)
            ]
        )

        # 创建共享专家网络的 ModuleList
        self.shared_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embedding_size * len(vocab_sizes), shared_units[0]),
                    nn.ELU(),
                    nn.Linear(shared_units[0], shared_units[1]),
                    nn.ELU(),
                )
                for _ in range(task_num)
            ]
        )

        # 创建输出层的 ModuleList
        self.output_layers = nn.ModuleList(
            [
                nn.Linear(task_specific_units[-1] + shared_units[-1], 1)
                for _ in range(task_num)
            ]
        )

    def forward(self, features: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """
        定义前向传播逻辑。

        参数:
            features (dict[int, torch.Tensor]): 输入特征的字典，键为特征 ID，值为对应的张量。

        返回:
            list[torch.Tensor]: 每个任务的输出张量列表。
        """
        # 获取嵌入表示并拼接
        embeddings = [
            self.embeddings[feat_id](features[feat_id]) for feat_id in features
        ]
        embeddings = torch.cat(embeddings, dim=-1)

        # 计算任务特定和共享输出
        task_outputs = []
        for i in range(self.task_num):
            specific_output = self.task_specific_experts[i][i](embeddings)
            shared_output = self.shared_experts[i](embeddings)
            combined_output = torch.cat([specific_output, shared_output], dim=-1)
            task_outputs.append(self.output_layers[i](combined_output))

        return task_outputs


def generate_random_features(vocab_sizes, batch_size):
    features = {}
    for feat_id, vocab_size in vocab_sizes.items():
        features[f"feat_{feat_id}"] = torch.randint(0, vocab_size, (batch_size,))
    return features


if __name__ == "__main__":
    # Hyperparameters
    vocab_sizes = {
        "101": 444720,
        "109_14": 12524,
        "110_14": 2981054,
        # Add other feature vocab sizes here...
    }
    embedding_size = 16
    task_specific_units = [128, 64]
    shared_units = [128, 64]
    task_num = 2
    batch_size = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = FDN(
        vocab_sizes, embedding_size, task_specific_units, shared_units, task_num
    )
    model.to(device)
    print(model)
    # Generate random features
    random_features = generate_random_features(vocab_sizes, batch_size)
    random_features = {k: v.to(device) for k, v in random_features.items()}

    # Forward pass
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        outputs = model(random_features)

    # Print outputs
    for i, output in enumerate(outputs):
        print(f"Task {i + 1} Output: {output.squeeze().cpu().numpy()}")
