import tensorflow as tf
from typing import Dict, Tuple


class InputLayer(tf.keras.layers.Layer):
    def __init__(self, feature_specs: Dict[str, Dict], **kwargs):
        super(InputLayer, self).__init__(**kwargs)
        self.feature_specs = feature_specs
        self.embedding_layers = {}

        # Create embedding layers for categorical features
        for feature_name, spec in feature_specs.items():
            if spec["type"] == "categorical":
                vocab_size = spec["vocab_size"]
                embedding_dim = spec.get(
                    "embedding_dim", 16
                )  # Default embedding dimension
                self.embedding_layers[feature_name] = tf.keras.layers.Embedding(
                    input_dim=vocab_size,
                    output_dim=embedding_dim,
                    name=f"{feature_name}_embedding",
                )

    def call(
        self, inputs: Dict[str, tf.Tensor]
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        embedded_features = {}
        numerical_features = []

        for feature_name, tensor in inputs.items():
            spec = self.feature_specs[feature_name]
            if spec["type"] == "categorical":
                if feature_name in self.embedding_layers:
                    embedded_features[feature_name] = self.embedding_layers[
                        feature_name
                    ](tensor)
                else:
                    # Handle cases where embedding is not needed or a different type of encoding
                    embedded_features[feature_name] = tf.cast(
                        tensor, tf.float32
                    )  # Example: cast to float
            elif spec["type"] == "numerical":
                numerical_features.append(tf.cast(tensor, tf.float32))
            # Add handling for other feature types like multi-hot if needed

        # Concatenate numerical features if any
        if numerical_features:
            numerical_features_concat = tf.concat(numerical_features, axis=-1)
        else:
            numerical_features_concat = None

        return numerical_features_concat, embedded_features

    def get_config(self):
        config = super(InputLayer, self).get_config()
        config.update(
            {
                "feature_specs": self.feature_specs,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
