import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_intermediate_fusion_model(
    tma_input_dim: int = 4096,
    tabular_input_dim: int = None,
    dropout: float = 0.3,
) -> keras.Model:
    if tabular_input_dim is None:
        raise ValueError("Tabular dimension cannot be None")

    # Inputs
    tma_input = keras.Input(shape=(tma_input_dim,), name="tma")
    tabular_input = keras.Input(shape=(tabular_input_dim,), name="tabular")

    # TMA encoder
    x = layers.Dense(256, activation="relu")(tma_input)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    tma_embedding = layers.Dense(64, activation="relu")(x)

    # Tabular encoder
    y = layers.Dense(128, activation="relu")(tabular_input)
    y = layers.Dropout(dropout)(y)
    tabular_embedding = layers.Dense(64, activation="relu")(y)

    # Fusion
    fused = layers.Concatenate(axis=1)([tma_embedding, tabular_embedding])

    # Classifier
    z = layers.Dense(64, activation="relu")(fused)
    z = layers.Dropout(dropout)(z)
    logits = layers.Dense(1)(z)
    output = layers.Reshape(())(logits)

    return keras.Model(inputs=[tma_input, tabular_input], outputs=output)
