import tensorflow as tf


def dirichlet_layer(evidence):
    alpha = evidence + 1
    S = tf.reduce_sum(alpha, axis=-1, keepdims=True)
    probs = alpha / S
    return probs, alpha, S


@tf.keras.saving.register_keras_serializable(package='energy_hmm')
class EvidentialLoss(tf.keras.losses.Loss):
    def __init__(self, annealing_rate=100.0, max_annealing_rate=0.2, name='evidential_loss'):
        super().__init__(name=name)
        self.annealing_rate     = float(annealing_rate)
        self.max_annealing_rate = float(max_annealing_rate)
        self.current_epoch      = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    def call(self, y_true, evidence):
        alpha   = evidence + 1
        S       = tf.reduce_sum(alpha, axis=-1, keepdims=True)
        probs   = alpha / S
        err     = tf.keras.losses.categorical_crossentropy(y_true, probs)
        var     = tf.reduce_sum(probs * (1 - probs) / (S + 1), axis=-1, keepdims=True)
        lambda_ = tf.minimum(self.max_annealing_rate, self.current_epoch / self.annealing_rate)
        return tf.reduce_mean(err + lambda_ * var, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'annealing_rate':     self.annealing_rate,
            'max_annealing_rate': self.max_annealing_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class AnnealingCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if hasattr(self.model, 'loss') and isinstance(self.model.loss, EvidentialLoss):
            self.model.loss.current_epoch.assign(tf.cast(epoch, tf.float32))


@tf.keras.saving.register_keras_serializable(package='energy_hmm')
def evidential_kl_divergence(y_true, y_pred):
    probs, _, _ = dirichlet_layer(y_pred)
    return tf.keras.losses.kullback_leibler_divergence(y_true, probs)


@tf.keras.saving.register_keras_serializable(package='energy_hmm')
def cross_entropy_loss(y_true, y_pred):
    probs, _, _ = dirichlet_layer(y_pred)
    return tf.keras.losses.categorical_crossentropy(y_true, probs)
