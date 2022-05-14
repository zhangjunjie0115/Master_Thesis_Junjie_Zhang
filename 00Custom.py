import tensorflow.compat.v2 as tf
from keras import backend


class Leveberg_Marquardt(Optimizer):

    def __init__(self, lr=1e-3, lambda_1=1e-5, lambda_2=1e+2, **kwargs):
        super(Leveberg_Marquardt, self).__init__(**kwargs)
        with backend.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = backend.variable(lr, name ='lr')
            self.lambda_1 = K.variable(lambda_1, name='lambda_1')
            self.lambda_2 = K.variable(lambda_2, name='lambda_2')

    def _create_all_weights(self, params):

        return

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [tf.compat.v1.assign_add(self.iterations, 1)]
        error = [K.int_shape(m) for m in loss]
        for p, g, err in zip(params, grads, error):
            H = K.dot(g, K.transpose(g)) + self.tau * K.eye(K.max(g))
            w = p - K.pow(H, -1) * K.dot(K.transpose(g), err)

            if self.lr and self.lambda_2 is not None:
                w = w - 1/self.tau * err
            if self.lr and self.lambda_1 is not None:
                w = w - K.pow(H, -1) * err
            if getattr(p, 'constraint', None) is not None:
                w = p.constraint(w)
            self.updates.append(K.update_add(err, w))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'lambda_1': float(K.get_value(self.lambda_1)),
                  'lambda_2': float(K.get_value(self.lambda_2)), }
        base_config = super(Leveberg_Marquardt, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
