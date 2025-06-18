from tensorflow.keras.optimizers import Adam


class SparseAdam(Adam):
    """
    SparseAdam is a variant of the Adam optimizer that is designed to work with sparse gradients.
    It inherits from the Adam optimizer in TensorFlow and can be used in the same way.
    """

    def __init__(
        self,
        embedding_learning_rate: float = 0.0075,
        dense_learning_rate: float = 0.00015,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_learning_rate = embedding_learning_rate
        self.dense_learning_rate = dense_learning_rate

    def _resource_apply_dense(self, grad, var, apply_state=None):
        """
        Applies dense gradients to the variable.
        """
        if "embedding" in var.name:
            return super()._resource_apply_dense(
                grad * self.embedding_learning_rate, var, apply_state
            )
        else:
            return super()._resource_apply_dense(
                grad * self.dense_learning_rate, var, apply_state
            )

        # Use the dense learning rate for dense variables

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        """
        Applies sparse gradients to the variable.
        """
        if "embedding" in var.name:
            return super()._resource_apply_sparse(
                grad * self.embedding_learning_rate, var, indices, apply_state
            )
        else:
            return super()._resource_apply_sparse(
                grad * self.dense_learning_rate, var, indices, apply_state
            )

    def get_config(self):
        """
        Returns the configuration of the SparseAdam optimizer.
        """
        config = super().get_config()
        config.update(
            {
                "embedding_learning_rate": self.embedding_learning_rate,
                "dense_learning_rate": self.dense_learning_rate,
            }
        )
        return config
