import tensorflow as tf
from tensorflow.keras import layers, Model

# -------------------------
# Generator
# -------------------------
def conv3(x, f, k=3, s=1, norm=True, act=True):
    """3D convolution block with optional batchnorm and activation."""
    x = layers.Conv3D(f, k, strides=s, padding="same")(x)
    if norm: x = layers.BatchNormalization()(x)
    if act: x = layers.LeakyReLU(0.2)(x)
    return x

def deconv3(x, f, k=3, s=2, norm=True, act=True):
    """3D transpose convolution block."""
    x = layers.Conv3DTranspose(f, k, strides=s, padding="same")(x)
    if norm: x = layers.BatchNormalization()(x)
    if act: x = layers.LeakyReLU(0.2)(x)
    return x

def build_generator(in_channels=3, out_channels=3, base=32):
    """Builds the 3D U-Net generator."""
    inp = layers.Input(shape=(None, None, None, in_channels))
    e1 = conv3(inp, base)
    p1 = conv3(e1, base, s=2)
    e2 = conv3(p1, base*2)
    p2 = conv3(e2, base*2, s=2)
    e3 = conv3(p2, base*4)
    p3 = conv3(e3, base*4, s=2)
    b  = conv3(p3, base*8)
    u3 = deconv3(b, base*4)
    u3 = layers.Concatenate()([u3, e3])
    u3 = conv3(u3, base*4)
    u2 = deconv3(u3, base*2)
    u2 = layers.Concatenate()([u2, e2])
    u2 = conv3(u2, base*2)
    u1 = deconv3(u2, base)
    u1 = layers.Concatenate()([u1, e1])
    u1 = conv3(u1, base)
    out = layers.Conv3D(out_channels, 1, padding="same", dtype='float32')(u1)
    return Model(inp, out, name="Generator")

# -------------------------
# Discriminator backbone
# -------------------------
def disc_backbone(x, base=32):
    """3D PatchGAN discriminator backbone."""
    feats = []
    x = layers.Conv3D(base, 4, 2, "same")(x); x = layers.LeakyReLU(0.2)(x); feats.append(x)
    x = layers.Conv3D(base*2, 4, 2, "same")(x); x = layers.BatchNormalization()(x); x = layers.LeakyReLU(0.2)(x); feats.append(x)
    x = layers.Conv3D(base*4, 4, 2, "same")(x); x = layers.BatchNormalization()(x); x = layers.LeakyReLU(0.2)(x); feats.append(x)
    x = layers.Conv3D(base*8, 4, 1, "same")(x); x = layers.BatchNormalization()(x); x = layers.LeakyReLU(0.2)(x); feats.append(x)
    logits = layers.Conv3D(1, 4, 1, "same", dtype='float32')(x)
    return logits, feats

def build_Ds(in_channels_hr=3, in_channels_cond=3, base=32):
    """Spatial discriminator."""
    inp = tf.keras.Input(shape=(None, None, None, in_channels_hr + in_channels_cond))
    logits, feats = disc_backbone(inp, base)
    return Model(inp, [logits, *feats], name="Ds")

def build_Dt(in_channels_hr=3, in_channels_cond=3, num_frames=2, base=32):
    """Temporal discriminator over two consecutive frames."""
    inp = tf.keras.Input(shape=(None, None, None, (in_channels_hr + in_channels_cond) * num_frames))
    logits, feats = disc_backbone(inp, base)
    return Model(inp, [logits, *feats], name="Dt")
