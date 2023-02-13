import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt

from typing import List
from utils import config

class ShiftedPatchTokenization(layers.Layer):
    """ 
    Class to handle the cropping and the patching of images

    methods:
    - crop_shifted_pad()
        creates padded images 

    - call()
        patchifies the given image and 

    """
    def __init__(
        self,
        image_size=config.IMAGE_SIZE,
        patch_size=config.PATCH_SIZE,
        num_patches=config.NUM_PATCHES,
        projection_dim=config.PROJECTION_DIM,
        vanilla=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.half_patch = patch_size // 2

        # valilla = original image
        self.vanilla = vanilla  

        # flatten the image patches
        self.flatten_patches = layers.Reshape((num_patches, -1))
        self.projection = layers.Dense(units=projection_dim)
        self.layer_norm = layers.LayerNormalization(epsilon=config.LAYER_NORM_EPS)

    def crop_shift_pad(self, images, mode):
        """ Creates padded images given a move direction. 
        Where the default value is "right-down".

        possible shift directions: 
        - "left-up"
        - "left-down"
        - "right-up"
        - "right-down"

        Args:
            images tf.Array: original image
            mode str: mode in which to shift the image

        Returns:
            _type_: _description_
        """

        crop_height, crop_width, shift_height, shift_width = 0, 0, 0, 0

        # Build the diagonally shifted images
        if mode == "left-up":
            crop_height = self.half_patch
            crop_width = self.half_patch

        elif mode == "left-down":
            crop_width = self.half_patch
            shift_height = self.half_patch

        elif mode == "right-up":
            crop_height = self.half_patch
            shift_width = self.half_patch

        else:
            shift_height = self.half_patch
            shift_width = self.half_patch

        # Crop the shifted images and pad them
        crop = tf.image.crop_to_bounding_box(
            images,
            offset_height=crop_height,
            offset_width=crop_width,
            target_height=self.image_size - self.half_patch,
            target_width=self.image_size - self.half_patch,
        )

        # create shifted images (adding boundry box)
        shift_pad = tf.image.pad_to_bounding_box(
            crop,
            offset_height=shift_height,
            offset_width=shift_width,
            target_height=self.image_size,
            target_width=self.image_size,
        )

        return shift_pad

    def call(self, images):

        if not self.vanilla:
            # Concat the shifted images with the original image
            images = tf.concat(
                [
                    images,
                    self.crop_shift_pad(images, mode="left-up"),
                    self.crop_shift_pad(images, mode="left-down"),
                    self.crop_shift_pad(images, mode="right-up"),
                    self.crop_shift_pad(images, mode="right-down"),
                ],
                axis=-1,
            )

        # Patchify the images and flatten it
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

        # flatten patches
        flat_patches = self.flatten_patches(patches)

        if not self.vanilla:
            # Layer normalize the flat patches and linearly project it
            tokens = self.layer_norm(flat_patches)
            tokens = self.projection(tokens)
        else:
            # Linearly project the flat patches
            tokens = self.projection(flat_patches)

        return (tokens, patches)


def visualizePatch(train_in: List[tf.Array]):
    # Get a random image from the training dataset
    # and resize the image
    image = train_in[np.random.choice(range(train_in.shape[0]))]
    resized_image = tf.image.resize(
        tf.convert_to_tensor([image]),
        size=(
            config.IMAGE_SIZE,
            config.IMAGE_SIZE
        )
    )

    # Vanilla patch maker: This takes an image and divides into
    # patches as in the original ViT paper
    (token, patch) = ShiftedPatchTokenization(vanilla=True)(resized_image / 255.0)
    (token, patch) = (token[0], patch[0])

    n = patch.shape[0]
    count = 1

    plt.figure(figsize=(4, 4))
    for row in range(n):
        for col in range(n):
            plt.subplot(n, n, count)
            count = count + 1
            image = tf.reshape(
                patch[row][col], (config.PATCH_SIZE, config.PATCH_SIZE, 3))

            plt.imshow(image)
            plt.axis("off")

    plt.show()

    # Shifted Patch Tokenization: This layer takes the image, shifts it
    # diagonally and then extracts patches from the concatinated images
    (token, patch) = ShiftedPatchTokenization(
        vanilla=False)(resized_image / 255.0)

    (token, patch) = (token[0], patch[0])
    n = patch.shape[0]

    shifted_images = ["ORIGINAL", "LEFT-UP", "LEFT-DOWN", "RIGHT-UP", "RIGHT-DOWN"]
    for index, shift_mode in enumerate(shifted_images):

        # print the type of shifted image
        print(shift_mode)

        count = 1
        plt.figure(figsize=(4, 4))

        for row in range(n):
            for col in range(n):

                plt.subplot(n, n, count)
                count = count + 1
                image = tf.reshape(
                    patch[row][col], (config.PATCH_SIZE, config.PATCH_SIZE, 5 * 3))

                plt.imshow(image[..., 3 * index : 3 * index + 3])
                plt.axis("off")

        plt.show()


class PatchEncoder(layers.Layer):

    def __init__(
        self, 
        num_patches=config.NUM_PATCHES,
        projection_dim=config.PROJECTION_DIM, 
        **kwargs
    ):
    
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

        self.positions = tf.range(start=0, limit=self.num_patches, delta=1)

    def call(self, encoded_patches):
        encoded_positions = self.position_embedding(self.positions)
        encoded_patches = encoded_patches + encoded_positions

        return encoded_patches
