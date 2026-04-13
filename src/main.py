import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# ─────────────────────────────────────────────
# 1. Copy model definition from vgg16.py
# ─────────────────────────────────────────────


class ConvBNRelu(tf.keras.Model):
    def __init__(
        self,
        filters,
        kernel_size=3,
        strides=1,
        padding="SAME",
        weight_decay=0.0005,
        rate=0.4,
        drop=True,
    ):
        super(ConvBNRelu, self).__init__()
        self.drop = drop
        self.conv = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        )
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.dropOut = keras.layers.Dropout(rate=rate)

    def call(self, inputs, training=False):
        layer = self.conv(inputs)
        layer = tf.nn.relu(layer)
        layer = self.batchnorm(layer, training=training)
        if self.drop:
            layer = self.dropOut(layer, training=training)
        return layer


class VGG16Model(tf.keras.Model):
    def __init__(self):
        super(VGG16Model, self).__init__()
        self.conv1 = ConvBNRelu(filters=64, kernel_size=[3, 3], rate=0.3)
        self.conv2 = ConvBNRelu(filters=64, kernel_size=[3, 3], drop=False)
        self.maxPooling1 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv3 = ConvBNRelu(filters=128, kernel_size=[3, 3])
        self.conv4 = ConvBNRelu(filters=128, kernel_size=[3, 3], drop=False)
        self.maxPooling2 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv5 = ConvBNRelu(filters=256, kernel_size=[3, 3])
        self.conv6 = ConvBNRelu(filters=256, kernel_size=[3, 3])
        self.conv7 = ConvBNRelu(filters=256, kernel_size=[3, 3], drop=False)
        self.maxPooling3 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv11 = ConvBNRelu(filters=512, kernel_size=[3, 3])
        self.conv12 = ConvBNRelu(filters=512, kernel_size=[3, 3])
        self.conv13 = ConvBNRelu(filters=512, kernel_size=[3, 3], drop=False)
        self.maxPooling5 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv14 = ConvBNRelu(filters=512, kernel_size=[3, 3])
        self.conv15 = ConvBNRelu(filters=512, kernel_size=[3, 3])
        self.conv16 = ConvBNRelu(filters=512, kernel_size=[3, 3], drop=False)
        self.maxPooling6 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flat = keras.layers.Flatten()
        self.dropOut = keras.layers.Dropout(rate=0.5)
        self.dense1 = keras.layers.Dense(
            units=512,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(0.0005),
        )
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.dropOut2 = keras.layers.Dropout(rate=0.5)  # fixed: was self.drop
        self.dense2 = keras.layers.Dense(units=10)
        self.softmax = keras.layers.Activation("softmax")

    def call(self, inputs, training=False):
        net = self.conv1(inputs, training=training)
        net = self.conv2(net, training=training)
        net = self.maxPooling1(net)
        net = self.conv3(net, training=training)
        net = self.conv4(net, training=training)
        net = self.maxPooling2(net)
        net = self.conv5(net, training=training)
        net = self.conv6(net, training=training)
        net = self.conv7(net, training=training)
        net = self.maxPooling3(net)
        net = self.conv11(net, training=training)
        net = self.conv12(net, training=training)
        net = self.conv13(net, training=training)
        net = self.maxPooling5(net)
        net = self.conv14(net, training=training)
        net = self.conv15(net, training=training)
        net = self.conv16(net, training=training)
        net = self.maxPooling6(net)
        net = self.dropOut(net, training=training)
        net = self.flat(net)
        net = self.dense1(net)
        net = self.batchnorm(net, training=training)
        net = self.dropOut2(net, training=training)  # fixed
        net = self.dense2(net)
        net = self.softmax(net)
        return net


# ─────────────────────────────────────────────
# 2. Load & normalise CIFAR-10
# ─────────────────────────────────────────────


def load_and_normalize():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images = train_images.astype(np.float32)
    test_images = test_images.astype(np.float32)
    mean = np.mean(train_images, axis=(0, 1, 2, 3))
    std = np.std(train_images, axis=(0, 1, 2, 3))
    train_images = (train_images - mean) / (std + 1e-7)
    test_images = (test_images - mean) / (std + 1e-7)
    test_labels = to_categorical(test_labels, 10)
    return test_images, test_labels, mean, std


# ─────────────────────────────────────────────
# 3. PGD Attack
# ─────────────────────────────────────────────


def pgd_attack(model, images, labels, epsilon=0.03, alpha=0.007, num_iter=40):
    """
    Projected Gradient Descent (Madry et al. 2018)

    Args:
        model    : your VGG16Model (inference mode)
        images   : clean input batch, shape (N, 32, 32, 3)
        labels   : one-hot labels,    shape (N, 10)
        epsilon  : max L-inf perturbation budget
        alpha    : step size per iteration
        num_iter : number of PGD steps

    Returns:
        adv_images : adversarial examples clipped to epsilon-ball
    """
    images = tf.cast(images, tf.float32)
    labels = tf.cast(labels, tf.float32)

    # Start from a random point inside the epsilon ball
    adv_images = images + tf.random.uniform(
        shape=tf.shape(images), minval=-epsilon, maxval=epsilon
    )

    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    for i in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(adv_images)
            preds = model(adv_images, training=False)
            loss = loss_fn(labels, preds)

        # Gradient w.r.t. the IMAGE (not the weights)
        gradients = tape.gradient(loss, adv_images)

        # FGSM step
        adv_images = adv_images + alpha * tf.sign(gradients)

        # Project back into epsilon-ball around original image
        adv_images = tf.clip_by_value(adv_images, images - epsilon, images + epsilon)

        if (i + 1) % 10 == 0:
            print(f"  PGD step {i+1}/{num_iter}  loss={loss.numpy():.4f}")

    return adv_images.numpy()


# ─────────────────────────────────────────────
# 4. Evaluate & visualise
# ─────────────────────────────────────────────


def evaluate(model, images, labels, tag=""):
    preds = model(images, training=False)
    acc = np.mean(np.argmax(preds.numpy(), axis=1) == np.argmax(labels, axis=1))
    print(f"[{tag}] Accuracy: {acc*100:.2f}%")
    return acc


def visualize(clean, adv, labels, model, mean, std, n=5):
    """Show clean vs adversarial side-by-side with predictions."""
    cifar10_classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    clean_preds = np.argmax(model(clean[:n], training=False).numpy(), axis=1)
    adv_preds = np.argmax(model(adv[:n], training=False).numpy(), axis=1)
    true_labels = np.argmax(labels[:n], axis=1)

    # Denormalize for display
    clean_disp = clean[:n] * std + mean
    adv_disp = adv[:n] * std + mean
    clean_disp = np.clip(clean_disp / 255.0, 0, 1)
    adv_disp = np.clip(adv_disp / 255.0, 0, 1)

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    for i in range(n):
        axes[0, i].imshow(clean_disp[i])
        axes[0, i].set_title(
            f"Clean\nTrue: {cifar10_classes[true_labels[i]]}\n"
            f"Pred: {cifar10_classes[clean_preds[i]]}",
            fontsize=8,
        )
        axes[0, i].axis("off")

        axes[1, i].imshow(adv_disp[i])
        axes[1, i].set_title(
            f"Adversarial\nTrue: {cifar10_classes[true_labels[i]]}\n"
            f"Pred: {cifar10_classes[adv_preds[i]]}",
            fontsize=8,
            color="red" if adv_preds[i] != true_labels[i] else "green",
        )
        axes[1, i].axis("off")

    plt.suptitle("Clean vs PGD Adversarial Examples (ε=0.03)", fontsize=13)
    plt.tight_layout()
    plt.savefig("pgd_results.png", dpi=150)
    plt.show()
    print("Saved pgd_results.png")


# ─────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    WEIGHTS_PATH = "models/cifar10vgg_custom.h5"
    EPSILON = 0.03  # perturbation budget (L-inf)
    ALPHA = 0.007  # step size
    NUM_ITER = 40  # PGD iterations
    N_SAMPLES = 100  # how many test images to attack

    # Load data
    print("Loading CIFAR-10...")
    test_images, test_labels, mean, std = load_and_normalize()
    x_test = test_images[:N_SAMPLES]
    y_test = test_labels[:N_SAMPLES]

    # Build & load model
    print("Building model and loading weights...")
    model = VGG16Model()
    model(x_test[:2], training=False)  # build graph first
    model.load_weights(WEIGHTS_PATH)
    print("Weights loaded.")

    # Baseline accuracy
    evaluate(model, x_test, y_test, tag="Clean")

    # Run PGD
    print(f"\nRunning PGD attack (ε={EPSILON}, α={ALPHA}, iters={NUM_ITER})...")
    x_adv = pgd_attack(
        model, x_test, y_test, epsilon=EPSILON, alpha=ALPHA, num_iter=NUM_ITER
    )

    # Adversarial accuracy
    evaluate(model, x_adv, y_test, tag="Adversarial")

    # Visualise
    visualize(x_test, x_adv, y_test, model, mean, std, n=5)
