import tensorflow as tf
import keras

def dirichlet_layer(evidence):
    alpha = evidence + 1
    S = tf.reduce_sum(alpha, axis=-1, keepdims=True)
    probs = alpha / S
    return probs, alpha, S


@keras.saving.register_keras_serializable(package='energy_hmm')
class EvidentialLoss(tf.keras.losses.Loss):
    def __init__(self, loss_type='mse', annealing_rate=100.0, max_annealing_rate=0.2, name='evidential_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        if loss_type not in ('mse', 'log', 'digamma', 'cross_entropy'):
            raise ValueError(f"loss_type must be 'mse', 'log', 'digamma', or 'cross_entropy', got '{loss_type}'")
        self.loss_type          = loss_type
        self.annealing_rate     = float(annealing_rate)
        self.max_annealing_rate = float(max_annealing_rate)
        self.current_epoch      = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    def _kl_divergence(self, alpha, num_classes):
        ones = tf.ones([1, num_classes], dtype=tf.float32)
        sum_alpha = tf.reduce_sum(alpha, axis=-1, keepdims=True)
        first_term = (
            tf.math.lgamma(sum_alpha)
            - tf.reduce_sum(tf.math.lgamma(alpha), axis=-1, keepdims=True)
            + tf.reduce_sum(tf.math.lgamma(ones), axis=-1, keepdims=True)
            - tf.math.lgamma(tf.reduce_sum(ones, axis=-1, keepdims=True))
        )
        second_term = tf.reduce_sum(
            (alpha - ones) * (tf.math.digamma(alpha) - tf.math.digamma(sum_alpha)),
            axis=-1, keepdims=True
        )
        return first_term + second_term

    def _annealed_kl(self, y_true, alpha, num_classes):
        annealing_coef = tf.minimum(self.max_annealing_rate, self.current_epoch / self.annealing_rate)
        kl_alpha = (alpha - 1) * (1 - y_true) + 1
        return annealing_coef * self._kl_divergence(kl_alpha, num_classes)

    def _mse_loss(self, y_true, alpha, S, num_classes):
        loglikelihood_err = tf.reduce_sum((y_true - (alpha / S)) ** 2, axis=-1, keepdims=True)
        loglikelihood_var = tf.reduce_sum(
            alpha * (S - alpha) / (S * S * (S + 1)), axis=-1, keepdims=True
        )
        return loglikelihood_err + loglikelihood_var + self._annealed_kl(y_true, alpha, num_classes)

    def _log_loss(self, y_true, alpha, S, num_classes):
        A = tf.reduce_sum(y_true * (tf.math.log(S) - tf.math.log(alpha)), axis=-1, keepdims=True)
        return A + self._annealed_kl(y_true, alpha, num_classes)

    def _digamma_loss(self, y_true, alpha, S, num_classes):
        A = tf.reduce_sum(y_true * (tf.math.digamma(S) - tf.math.digamma(alpha)), axis=-1, keepdims=True)
        return A + self._annealed_kl(y_true, alpha, num_classes)

    def call(self, y_true, evidence):
        if self.loss_type == 'cross_entropy':
            probs = evidence
            return tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(y_true, probs)
            )

        alpha = evidence + 1
        S = tf.reduce_sum(alpha, axis=-1, keepdims=True)
        num_classes = tf.shape(alpha)[-1]

        if self.loss_type == 'mse':
            loss = self._mse_loss(y_true, alpha, S, num_classes)
        elif self.loss_type == 'log':
            loss = self._log_loss(y_true, alpha, S, num_classes)
        else:
            loss = self._digamma_loss(y_true, alpha, S, num_classes)

        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            'loss_type':            self.loss_type,
            'annealing_rate':       self.annealing_rate,
            'max_annealing_rate':   self.max_annealing_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class AnnealingCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if hasattr(self.model, 'loss') and isinstance(self.model.loss, EvidentialLoss):
            self.model.loss.current_epoch.assign(tf.cast(epoch, tf.float32))


@keras.saving.register_keras_serializable(package='energy_hmm')
def evidential_kl_divergence(y_true, y_pred):
    probs, _, _ = dirichlet_layer(y_pred)
    return tf.keras.losses.kullback_leibler_divergence(y_true, probs)


@keras.saving.register_keras_serializable(package='energy_hmm')
def cross_entropy_loss(y_true, y_pred):
    probs, _, _ = dirichlet_layer(y_pred)
    return tf.keras.losses.categorical_crossentropy(y_true, probs)
