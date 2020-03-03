import numpy as np
from layers import Dense , Activations
# from activation import Sigmoid,Swish
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
    swish1 = Activations('Swish')
    dense2 = Dense(2,W2,b2)
    swish2 = Activations('Swish')
    activation2 = Activations('Sigmoid')
    
    loss_fn = MSE()

    z1 = dense.forward(x)
    sig1 = sigmoid.forward(z1)
    z2 = dense2.forward(sig1)
    y_pred = activation2.forward(z2)

    sw1 = swish1.forward(z1)
    sw2 = dense2.forward(sw1)
    y_pre = swish2.forward(sw2)

    # loss = loss_fn.loss(y_true, y_pred)
    # print("loss: ", loss)
    # print("loss's mean: ",np.mean(loss))

    # sigloss = loss_fn.loss(y_true, y_pre)
    # print("SIG loss: ", sigloss)
    # print("sig loss's mean: ",np.mean(sigloss))

    # dldy_pred = loss_fn.gradient(y_true , y_pred)
    # print("lldy: ",dldy_pred)
    # dldz2 = activation2.backward(dldy_pred)
    # print("dldz2: ",dldz2)
    # dLda1 = dense2.backward(dldz2)
    # print("dLda1: ",dLda1)
    # dLz1 = sigmoid.backward(dLda1)
    # dLdw = dense.backward(dLz1)
    

    d1 =loss_fn.gradient(y_true , y_pre)
    # a = swish2.gradient(d1)
    # print(d1)
    d2 = swish2.backward(d1)
    d3 = dense2.backward(d2)
    d4 = swish1.backward(d3)
    d5 = dense.backward(d4)
    print(d2)





    