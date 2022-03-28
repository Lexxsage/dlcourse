import numpy as np

def softmax(predictions):
    if (predictions.ndim == 1):
        predictionsTmp = np.copy(predictions)
        predictionsTmp -= np.max(predictions)
        divider = 0;
        for i in range(len(predictions)):
            divider += np.exp(predictionsTmp[i])
        probs = np.exp(predictionsTmp)/divider
        return probs
    else:
        predictionsTmp = np.copy(predictions)
        predictionsTmp = (predictionsTmp.T - np.max(predictions,axis = 1)).T
        exp_pred = np.exp(predictionsTmp)
        exp_sum= np.sum(exp_pred,axis=1)
        return (exp_pred.T / exp_sum).T


def cross_entropy_loss(probs, target_index):
    if (probs.ndim == 1):
        loss = -np.log(probs[target_index])
    else:
        batch_size = probs.shape[0]
        loss_arr = -np.log(probs[range(batch_size), target_index])
        loss = np.sum(loss_arr) / batch_size

    return loss

def l2_regularization(W, reg_strength):
    loss = reg_strength * np.sum(np.square(W))
    grad = 2 * reg_strength * W

    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    dprediction = probs
    if (probs.ndim == 1):
        dprediction[target_index] -= 1
    else:
        batch_size = predictions.shape[0]
        dprediction[range(batch_size), target_index] -= 1
        dprediction /= batch_size
    return loss, dprediction



class Param:
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X
        res = X.copy()
        for x in np.nditer(res, op_flags=['readwrite']) :
            x[...] = x if x>=0 else 0
        return res

    def backward(self, d_out):
        res = self.X.copy()
        for x in np.nditer(res, op_flags=['readwrite']):
            x[...] = 1 if x>=0 else 0
        return res*d_out

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return np.dot(X,self.W.value)+self.B.value

    def backward(self, d_out):
        d_input = np.dot(d_out, self.W.value.transpose())
        self.W.grad += np.dot(self.X.transpose(), d_out)
        self.B.grad += np.dot(np.ones((1, d_out.shape[0])), d_out)

        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer

        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))
        self.X = None
        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape

        res_height  = 0
        res_width = 0
        #
        X_with_pad = np.zeros((batch_size , height + 2*self.padding , width + 2*self.padding , channels))
        X_with_pad[:, self.padding: height + self.padding, self.padding: width + self.padding, :] = X
        self.X = X_with_pad

        res_height  = X_with_pad.shape[1] - self.filter_size + 1
        res_width = X_with_pad.shape[2] - self.filter_size + 1
        out = np.zeros((batch_size , res_height  , res_width , self.out_channels))
        #

        for y in range(res_height ):
            for x in range(res_width):
                #
                X_matrix = X_with_pad[:, y: y + self.filter_size, x:x + self.filter_size, :]
                X_matrix_arr = X_matrix.reshape(batch_size, self.filter_size*self.filter_size * self.in_channels)
                W_arr = self.W.value.reshape(self.filter_size * self.filter_size * self.in_channels, self.out_channels)
                Res_arr = np.dot(X_matrix_arr , W_arr) + self.B.value
                Res_mat = Res_arr.reshape(batch_size, 1, 1, self.out_channels)
                out[: , y: y + self.filter_size , x:x + self.filter_size, :] = Res_mat
                #
        return out


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, res_height, res_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        #
        dX = np.zeros((batch_size, height, width, channels))
        W_arr = self.W.value.reshape(self.filter_size * self.filter_size * self.in_channels, self.out_channels)
        for x in range(res_width):
            for y in range(res_height ):
                X_matrix = self.X[:, x:x + self.filter_size , y:y + self.filter_size, :]
                X_matrix_arr = X_matrix.reshape(batch_size, self.filter_size * self.filter_size * self.in_channels)
                d_local = d_out[:, x:x + 1, y:y + 1, :]
                dX_arr = np.dot(d_local.reshape(batch_size, -1), W_arr.T)
                dX[:, x:x +self.filter_size , y:y+self.filter_size, :] += dX_arr.reshape(X_matrix.shape)
                dW = np.dot(X_matrix_arr.T, d_local.reshape(batch_size, -1))
                dB = np.dot(np.ones((1, d_local.shape[0])), d_local.reshape(batch_size, -1))
                self.W.grad += dW.reshape(self.W.value.shape)
                self.B.grad += dB.reshape(self.B.value.shape)
        return dX[:, self.padding : (height - self.padding), self.padding : (width - self.padding), :]
        #

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool
        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None
        self.masks = {}

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X
        self.masks.clear()

        res_height  = int((height - self.pool_size) / self.stride + 1)
        res_width = int((width - self.pool_size) / self.stride + 1)

        output = np.zeros((batch_size, res_height, res_width, channels))

        mult = self.stride

        for x in range(res_width):
            for y in range(res_height):
                I = X[:, x*mult:x*mult + self.pool_size, y*mult:y*mult + self.pool_size, :]
                self.mask(x=I, pos=(x, y) )
                output[:, x, y, :] = np.max(I, axis=(1, 2))
        return output

    def backward(self, d_out):
        _, res_height, res_width, _ = d_out.shape
        dX = np.zeros_like(self.X)

        mult = self.stride


        for x in range(res_width):
            for y in range(res_height):

                dX[:, x*mult:x*mult+self.pool_size, y*mult:y*mult+self.pool_size, :] += d_out[:, x:x + 1, y:y + 1, :] * self.masks[(x, y)]
        return dX

    def mask(self, x, pos):
        zero_mask = np.zeros_like(x)
        batch_size, height, width, channels = x.shape
        x = x.reshape(batch_size, height * width, channels)
        idx = np.argmax(x, axis=1)

        n_idx, c_idx = np.indices((batch_size, channels))
        zero_mask.reshape(batch_size, height * width, channels)[n_idx, idx, c_idx] = 1
        self.masks[pos] = zero_mask

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = X.shape
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        #ok
        return {}