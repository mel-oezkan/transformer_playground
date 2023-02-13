import math

import tensorflow as tf
from tensorflow.keras import layers

from utils import config

class MultiHeadAtentionLSA(tf.keras.layers.MultiHeadAttention):
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


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def create_vit_classifier(diag_attn_mask, vanilla=False):
    inputs = layers.Input(shape=config.INPUT_SHAPE)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    (tokens, _) = ShiftedPatchTokenization(vanilla=vanilla)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder()(tokens)

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

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=config.MLP_HEAD_UNITS, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(config.NUM_CLASSES)(features)
    
    
    model = keras.Model(inputs=inputs, outputs=logits)
    return model