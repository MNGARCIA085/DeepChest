# losses/weighted_bce.py
import tensorflow as tf

def weighted_bce(pos_weights):
    def loss(y_true, logits): # y_pred insetad of logits
        loss = tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true,
            logits=logits, # y_pred
            pos_weight=pos_weights
        )
        return tf.reduce_mean(loss) # is this really neede, chcek later!!!!!!!!!!!
    return loss




