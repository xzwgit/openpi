"""Native JAX training-time image augmentation.

This module replaces the abandoned `augmax` dependency (0.4.1), which breaks
under jax >= 0.10: augmentation had to be `vmap`-ed over the (sharded) batch
axis, and jax >= 0.10 rejects vmap when the mapped-away axes of the inputs
are sharded inconsistently (e.g. images vs. PRNG keys).

The implementation here is therefore fully vectorized over the batch: every
op works directly on `[B, H, W, C]` arrays with per-sample random parameters,
no vmap involved, so it composes with any sharding under jit.

The augmentation family and its semantics mirror the PyTorch training path in
`openpi/models_pytorch/preprocessing_pytorch.py` (which in turn matched the
original augmax chain `RandomCrop(0.95w, 0.95h) + Resize + Rotate(-5, 5) +
ColorJitter(0.3, 0.4, 0.5)` on images in [0, 1]):

  * geometric (non-wrist cameras only):
      - random crop covering ~95% of the image, resized back with bilinear
        interpolation (half-pixel convention, like
        `F.interpolate(..., mode="bilinear", align_corners=False)`)
      - random rotation by U(-5, 5) degrees with zero padding (like
        `grid_sample(..., padding_mode="zeros", align_corners=False)`)
  * color (all cameras), applied in order per sample:
      - brightness: image * U(0.7, 1.3)
      - contrast:   (image - mean) * U(0.6, 1.4) + mean   (per-sample mean)
      - saturation: gray + (image - gray) * U(0.5, 1.5),
        where gray is the channel-wise mean
      - clamp to [0, 1]

All functions are stateless, consume explicit PRNG keys, and operate on
float `[B, H, W, C]` images with values in [0, 1]. Random parameters are
drawn per batch element (and, in `preprocess_observation`, per camera key);
the PyTorch path draws once per call for the whole batch - the distributions
are identical, ours are just applied per-sample like the old augmax setup.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = [
    "train_image_augment",
    "crop_resize",
    "rotate_by",
    "color_jitter",
]


def _gather_yx(images: jax.Array, y_idx: jax.Array, x_idx: jax.Array) -> jax.Array:
    """Gather `images` [B, H, W, C] at per-sample [B, H, W] integer coordinates.

    Uses a single `take_along_axis` gather over the flattened (H, W) axis
    instead of advanced indexing on the batch axis: sample b only reads its
    own pixels, so the op composes with any sharding of the batch axis under
    jax >= 0.10 (which cannot resolve the output sharding of a batch-axis
    gather). Two chained per-axis gathers would be wrong because the row index
    must stay the one evaluated at the *output* column, not the gathered one.
    """
    b, h, w, c = images.shape
    flat_idx = (y_idx * w + x_idx)[..., None]  # [B, H, W, 1]
    gathered = jnp.take_along_axis(images.reshape(b, h * w, c), flat_idx.reshape(b, h * w, 1), axis=1)
    return gathered.reshape(b, h, w, c)


def _bilinear_sample_zeros(images: jax.Array, src_y: jax.Array, src_x: jax.Array) -> jax.Array:
    """Bilinearly sample `images` [B, H, W, C] at float pixel coordinates.

    `src_y`, `src_x` are [B, H, W] source coordinates. Exactly matches
    `grid_sample(mode="bilinear", padding_mode="zeros", align_corners=False)`:
    out-of-bounds neighbor pixels contribute zero, so partially-out-of-bounds
    coordinates blend in zeros by weight.
    """
    h, w = images.shape[1], images.shape[2]
    y0f = jnp.floor(src_y)
    x0f = jnp.floor(src_x)
    wy = src_y - y0f
    wx = src_x - x0f
    y0 = jnp.clip(y0f, 0.0, h - 1.0).astype(jnp.int32)
    x0 = jnp.clip(x0f, 0.0, w - 1.0).astype(jnp.int32)
    y1 = jnp.clip(y0f + 1.0, 0.0, h - 1.0).astype(jnp.int32)
    x1 = jnp.clip(x0f + 1.0, 0.0, w - 1.0).astype(jnp.int32)
    # Validity of each of the 4 neighbors: index i is in-range iff 0 <= i <= size - 1.
    y0_ok = (y0f >= 0.0) & (y0f <= h - 1.0)
    y1_ok = (y0f + 1.0 >= 0.0) & (y0f + 1.0 <= h - 1.0)
    x0_ok = (x0f >= 0.0) & (x0f <= w - 1.0)
    x1_ok = (x0f + 1.0 >= 0.0) & (x0f + 1.0 <= w - 1.0)
    c00 = jnp.where((y0_ok & x0_ok)[..., None], _gather_yx(images, y0, x0), 0.0)
    c01 = jnp.where((y0_ok & x1_ok)[..., None], _gather_yx(images, y0, x1), 0.0)
    c10 = jnp.where((y1_ok & x0_ok)[..., None], _gather_yx(images, y1, x0), 0.0)
    c11 = jnp.where((y1_ok & x1_ok)[..., None], _gather_yx(images, y1, x1), 0.0)
    top = c00 * (1.0 - wx)[..., None] + c01 * wx[..., None]
    bottom = c10 * (1.0 - wx)[..., None] + c11 * wx[..., None]
    return top * (1.0 - wy)[..., None] + bottom * wy[..., None]


def crop_resize(images: jax.Array, start_h: jax.Array, start_w: jax.Array) -> jax.Array:
    """Crop `int(0.95 * size)` windows at per-sample offsets and resize back.

    `start_h`/`start_w` are per-sample [B] integer offsets. Mirrors
    `image[:, :, s_h:s_h+c_h, s_w:s_w+c_w, :]` followed by bilinear
    `F.interpolate(..., align_corners=False)` in the PyTorch path, including
    its half-pixel source coordinates and edge handling *within the crop*
    (negative source coordinates clamp to 0, and the last crop row/column
    replicates instead of reading past the crop window).
    """
    h, w = images.shape[1], images.shape[2]
    crop_h, crop_w = int(h * 0.95), int(w * 0.95)
    # Crop-relative half-pixel source coordinates (torch clamps negatives to 0).
    ys = jnp.arange(h, dtype=images.dtype)
    xs = jnp.arange(w, dtype=images.dtype)
    src_y = jnp.maximum((ys[None, :, None] + 0.5) * (crop_h / h) - 0.5, 0.0)  # [1, H, 1]
    src_x = jnp.maximum((xs[None, None, :] + 0.5) * (crop_w / w) - 0.5, 0.0)  # [1, 1, W]
    y0f = jnp.floor(src_y)
    x0f = jnp.floor(src_x)
    wy = src_y - y0f
    wx = src_x - x0f
    # Neighbor indices are clamped to the crop extent, then shifted by the
    # per-sample crop offset (so they never leave the crop window).
    y0 = jnp.clip(y0f, 0.0, crop_h - 1.0).astype(jnp.int32) + start_h[:, None, None]  # [B, H, 1]
    y1 = jnp.clip(y0f + 1.0, 0.0, crop_h - 1.0).astype(jnp.int32) + start_h[:, None, None]
    x0 = jnp.clip(x0f, 0.0, crop_w - 1.0).astype(jnp.int32) + start_w[:, None, None]  # [B, 1, W]
    x1 = jnp.clip(x0f + 1.0, 0.0, crop_w - 1.0).astype(jnp.int32) + start_w[:, None, None]
    y0, y1, x0, x1 = jnp.broadcast_arrays(y0, y1, x0, x1)  # all [B, H, W]
    top = _gather_yx(images, y0, x0) * (1.0 - wx)[..., None] + _gather_yx(images, y0, x1) * wx[..., None]
    bottom = _gather_yx(images, y1, x0) * (1.0 - wx)[..., None] + _gather_yx(images, y1, x1) * wx[..., None]
    return top * (1.0 - wy)[..., None] + bottom * wy[..., None]


def random_crop_resize(images: jax.Array, key: jax.Array) -> jax.Array:
    """`crop_resize` with uniformly sampled per-sample offsets (like `torch.randint(0, max + 1)`)."""
    b, h, w = images.shape[0], images.shape[1], images.shape[2]
    key_h, key_w = jax.random.split(key)
    start_h = jax.random.randint(key_h, (b,), 0, h - int(h * 0.95) + 1)
    start_w = jax.random.randint(key_w, (b,), 0, w - int(w * 0.95) + 1)
    return crop_resize(images, start_h, start_w)


def rotate_by(images: jax.Array, angle_deg: jax.Array) -> jax.Array:
    """Rotate each image by `angle_deg` degrees (per-sample [B] or scalar), zero padding.

    Uses the same normalized-grid convention as the PyTorch path
    (`grid_sample(..., align_corners=False)`) so results agree numerically.
    """
    b, h, w = images.shape[0], images.shape[1], images.shape[2]
    angle_deg = jnp.broadcast_to(jnp.asarray(angle_deg, dtype=images.dtype), (b,))
    angle = angle_deg.reshape(b, 1, 1) * (jnp.pi / 180.0)
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    # Normalized pixel-center grid in [-1, 1].
    gy = jnp.linspace(-1.0, 1.0, h, dtype=images.dtype)
    gx = jnp.linspace(-1.0, 1.0, w, dtype=images.dtype)
    yy, xx = jnp.meshgrid(gy, gx, indexing="ij")  # [H, W]
    # For every output pixel, sample the source at the rotated location.
    src_nx = xx[None] * cos_a - yy[None] * sin_a  # [B, H, W]
    src_ny = xx[None] * sin_a + yy[None] * cos_a
    # align_corners=False denormalization: [-1, 1] -> [-0.5, size - 0.5].
    src_x = (src_nx + 1.0) * (w / 2.0) - 0.5
    src_y = (src_ny + 1.0) * (h / 2.0) - 0.5
    return _bilinear_sample_zeros(images, src_y, src_x)


def rotate(images: jax.Array, key: jax.Array) -> jax.Array:
    """Rotate each image by a random angle in [-5, 5] degrees."""
    b = images.shape[0]
    angle_deg = jax.random.uniform(key, (b,), minval=-5.0, maxval=5.0)
    return rotate_by(images, angle_deg)


def color_jitter_factors(key: jax.Array, batch_size: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Draw per-sample (brightness, contrast, saturation) factors, each [B, 1, 1, 1].

    Matches both the augmax parameters (jitter strength 0.3/0.4/0.5 around 1)
    and the PyTorch path (U(0.7, 1.3) / U(0.6, 1.4) / U(0.5, 1.5)).
    """
    key_b, key_c, key_s = jax.random.split(key, 3)
    shape = (batch_size, 1, 1, 1)
    brightness = jax.random.uniform(key_b, shape, minval=0.7, maxval=1.3)
    contrast = jax.random.uniform(key_c, shape, minval=0.6, maxval=1.4)
    saturation = jax.random.uniform(key_s, shape, minval=0.5, maxval=1.5)
    return brightness, contrast, saturation


def color_jitter(images: jax.Array, key: jax.Array) -> jax.Array:
    """Apply per-sample brightness, contrast and saturation jitter, then clamp to [0, 1]."""
    brightness, contrast, saturation = color_jitter_factors(key, images.shape[0])
    images = images * brightness
    mean = jnp.mean(images, axis=(1, 2, 3), keepdims=True)
    images = (images - mean) * contrast + mean
    gray = jnp.mean(images, axis=-1, keepdims=True)
    images = gray + (images - gray) * saturation
    return jnp.clip(images, 0.0, 1.0)


def train_image_augment(images: jax.Array, key: jax.Array, *, geometric: bool = True) -> jax.Array:
    """Full training augmentation for a [B, H, W, C] batch with values in [0, 1].

    `geometric=False` skips crop/rotation (used for wrist cameras, as in both
    the augmax and PyTorch paths). All random parameters are drawn per batch
    element; the same `key` always reproduces the same output.
    """
    key_geo, key_color = jax.random.split(key)
    if geometric:
        key_crop, key_rot = jax.random.split(key_geo)
        images = random_crop_resize(images, key_crop)
        images = rotate(images, key_rot)
    return color_jitter(images, key_color)
