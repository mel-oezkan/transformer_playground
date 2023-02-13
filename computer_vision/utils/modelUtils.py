import math

import tensorflow as tf
from tensorflow.keras import layers

from utils import config
from utils import patchUtils

class MultiHeadAttentionLSA(tf.keras.layers.MultiHeadAttention):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # trainable tau value   
        self.tau = tf.Variable(
            math.sqrt(float(self._key_dim)), trainable=True
        )


    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        """Computes attention weights and scores"""

        query = tf.multiply(query, 1.0 / self.tau)
        
        attention_scores = tf.einsum(self._dot_product_equation, key, query)
        attention_scores = self._masked_softmax(attention_scores, attention_mask)
        attention_scores_dropout = self._dropout_layer(
            attention_scores, training=training
        )

        attention_output = tf.einsum(
            self._combine_equation, attention_scores_dropout, value
        )
        
        return attention_output, attention_scores


def mlp(x, hidden_units: int, dropout_rate: float):
    """Creates a simple mlp block.

    Args:
        x (tf.Array): input to dense layer 
        hidden_units (int): number of hidden units
        dropout_rate (float): dropout rate

    Returns:
        _type_: _description_
    """

    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def data_augmentation(x_train: tf.Tensor):
    """Create data augmentation pipeline trough a sequential model.

    Args:
    
        x (List[tf.Array]): train images for alignment

    Returns:
        tf.keras.layers: augmentation layer
    """

    # augment data
    data_augmentation = tf.keras.Sequential(
        [
            layers.Normalization(),
            layers.Resizing(
                config.IMAGE_SIZE,
                config.IMAGE_SIZE),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        ],
        name="data_augmentation",
    )

    # Compute the mean and the variance of the training data for normalization.
    data_augmentation.layers[0].adapt(x_train)
    return data_augmentation

def create_vit_classifier(x_train, vanilla=False):
    """Combines the patching, the encoding and the transformer 
    layers and returns the keras model. Requires the train data to 
    adapt the data augmentation layer.
    
    Args:
        x_train (tf.Array): training images for augmentation layer 
        vanilla (bool, optional): Defines if the images are vanilla or shifted. Defaults to False.

    Returns:
        tf.keras.Model: keras model 
    """
    # Build the diagonal attention mask for later
    diag_attn_mask = 1 - tf.eye(config.NUM_PATCHES)
    diag_attn_mask = tf.cast([diag_attn_mask], dtype=tf.int8)

    # Augment data.
    inputs = layers.Input(shape=config.INPUT_SHAPE)
    augmented = data_augmentation(x_train)(inputs)
    
    # create and encode patches.
    (tokens, _) = patchUtils.ShiftedPatchTokenization(vanilla=vanilla)(augmented)
    encoded_patches = patchUtils.PatchEncoder()(tokens)

    # Create multiple layers of the Transformer block.
    for _ in range(config.TRANSFORMER_LAYERS):

        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Create a multi-head attention layer.
        if not vanilla:
            attention_output = MultiHeadAttentionLSA(
                num_heads=config.NUM_HEADS, key_dim=config.PROJECTION_DIM, dropout=0.1
            )(x1, x1, attention_mask=diag_attn_mask)
        else:
            attention_output = layers.MultiHeadAttention(
                num_heads=config.NUM_HEADS,
                key_dim=config.PROJECTION_DIM,
                dropout=0.1
            )(x1, x1)

        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=config.TRANSFORMER_UNITS, dropout_rate=0.1)

        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    
    # add MLP and classification layer.
    features = mlp(representation, hidden_units=config.MLP_HEAD_UNITS, dropout_rate=0.5)
    logits = layers.Dense(config.NUM_CLASSES)(features)
    
    return tf.keras.Model(inputs=inputs, outputs=logits)