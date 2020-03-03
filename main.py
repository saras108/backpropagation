import numpy as np
from layers import Dense , Activations
from activation import Sigmoid
from losses import MSE

if __name__ == "__main__":
    x = np.array([[.05,.1]])
    W1 = np.array([
        [0.15,0.20],
        [0.25,0.30]
        ])
    b1 = 0.35

    W2 = np.array([
        [0.4,0.45],
        [0.50,0.55]
    ])
    b2 =0.60

    y_true = np.array([[0.01, 0.99]])

    dense = Dense(2,W1,b1)
    sigmoid = Activations('Sigmoid')
    dense2 = Dense(2,W2,b2)
    activation2 = Activations('Sigmoid')
    loss_fn = MSE()

    z1 = dense.forward(x)
    # print(z1)
    sig1 = sigmoid.forward(z1)
    # print(sig1)
    z2 = dense2.forward(sig1)
    y_pred = activation2.forward(z2)

    loss = loss_fn.loss(y_true, y_pred)
    print("loss: ", loss)
    print("loss's mean: ",np.mean(loss))

    dldy_pred = loss_fn.gradient(y_true , y_pred)
    print("lldy: ",dldy_pred)

    dldz2 = activation2.backward(dldy_pred)
    

    a = dense2.backward(dldz2)
    b = sigmoid.backward(a)
    c = dense.backward(b)