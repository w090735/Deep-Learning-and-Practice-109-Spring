import numpy as np
import matplotlib.pyplot as plt

def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
          continue
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def derivative_sigmoid(sigmoid_value):
    return sigmoid_value * (1.0 - sigmoid_value)

# cross entropy loss
def loss(y, pred_y):
    return -(1 / y.shape[0]) * (np.matmul(y.T, np.log(pred_y + 0.001).T) + np.matmul((1 - y).T, np.log(1 - pred_y + 0.001).T))

# x --linear--> a1 --sigmoid--> z1 --linear--> a2 --sigmoid--> z2 --linear--> a3 --sigmoid--> pred_y
def forward():
    output["a1"] = np.matmul(w1, x.T)
    output["z1"] = sigmoid(output["a1"])
    output["a2"] = np.matmul(w2, output["z1"])
    output["z2"] = sigmoid(output["a2"])
    output["a3"] = np.matmul(w3, output["z2"])
    pred_y = sigmoid(output["a3"])

    return pred_y

# @: matrix multiply
# *: element-wise multiply
# dL/dz_i = w_(i+1) @ dL/da_(i+1)
# dL/da_i = dL/dz_i * sigmoid'(z_i)
# dL/dw_i = dL/da_i @ z_(i-1)
# z_(i-1) <--w_i-- a_i <- z_i <--w_(i+1)-- a_(i+1)
def backward(w1, w2, w3, pred_y):
    dev_L["y"] = -(y.T / (pred_y + 0.001) - (1 - y.T) / (1 - pred_y + 0.001))
    dev_L["a3"] = dev_L["y"] * derivative_sigmoid(pred_y)
    dev_L["w3"] = np.matmul(dev_L["a3"], output["z2"].T) * (1 / y.shape[0])

    dev_L["z2"] = np.matmul(w3.T, dev_L["a3"])
    dev_L["a2"] = dev_L["z2"] * derivative_sigmoid(output["z2"])
    dev_L["w2"] = np.matmul(dev_L["a2"], output["z1"].T) * (1 / y.shape[0])

    dev_L["z1"] = np.matmul(w2.T, dev_L["a2"])
    dev_L["a1"] = dev_L["z1"] * derivative_sigmoid(output["z1"])
    dev_L["w1"] = np.matmul(dev_L["a1"], x)

    w1 -= lamda * dev_L["w1"]
    w2 -= lamda * dev_L["w2"]
    w3 -= lamda * dev_L["w3"]

# acc = correct / total
def accuracy(y, pred_y):
    correct = 0
    result = np.zeros(pred_y.shape)
    result[pred_y >= 0.5] = 1
    result[pred_y < 0.5] = 0
    for i in range(y.shape[0]):
        if result[0][i] == y[i]:
            correct += 1
    return correct/x.shape[0]

def show_result(x, y, pred_y):
    result = np.zeros(pred_y.shape)
    result[pred_y >= 0.5] = 1
    result[pred_y < 0.5] = 0
    print(f'\npredict result:\n')
    for i in range(pred_y.shape[1]):
        print(pred_y[0][i])
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if result[0][i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()

def linear_exp():
    # training data
    x, y = generate_linear(n=100)
    # initialize learning rate
    lamda = 0.1

    # initialize weights
    w1 = np.random.uniform(0, 1, (3, 2))
    w2 = np.random.uniform(0, 1, (3, 3))
    w3 = np.random.uniform(0, 1, (1, 3))

    # initialize forward pass result and gradient
    output = {"a1":np.zeros((3, x.shape[0])), "z1":np.zeros((3, x.shape[0])), "a2":np.zeros((3, x.shape[0])), "z2":np.zeros((3, x.shape[0])),
              "a3":np.zeros((1, x.shape[0]))}
    dev_L = {"y":np.zeros((1, x.shape[0])), "a3":np.zeros((1, x.shape[0])), "z2":np.zeros((3, x.shape[0])), "a2":np.zeros((3, x.shape[0])),
             "z1":np.zeros((3, x.shape[0])), "a1":np.zeros((3, x.shape[0])), "w3":np.zeros((1, 3)), "w2":np.zeros((3, 3)), "w1":np.zeros((3, 2))}

    # training process
    acc = 0.0
    pre_acc = 0.0
    history = list()
    final = 0
    for i in range(30000):
        pred_y = forward()
        L = loss(y, pred_y)
        history.append(L[0][0])
        acc = accuracy(y, pred_y)
        if (i % 200) == 0 or acc == 1.0:
            print("---------------------")
            print(f'epoch: {i+1}, loss: {L[0][0]}, acc: {acc}')
        if acc == 1.0 or i == 30000-1:
            final = i
            break
        backward(w1, w2, w3, pred_y)

    # show record
    show_result(x, y, pred_y)
    plt.title("Learning Curve")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(range(final+1),history)

def XOR_exp():
    # training data
    x, y = generate_XOR_easy()
    # initialize learning rate
    lamda = 0.1

    # initialize weights
    w1 = np.random.uniform(0, 1, (3, 2))
    w2 = np.random.uniform(0, 1, (3, 3))
    w3 = np.random.uniform(0, 1, (1, 3))

    # initialize forward pass result and gradient
    output = {"a1":np.zeros((3, x.shape[0])), "z1":np.zeros((3, x.shape[0])), "a2":np.zeros((3, x.shape[0])), "z2":np.zeros((3, x.shape[0])),
              "a3":np.zeros((1, x.shape[0]))}
    dev_L = {"y":np.zeros((1, x.shape[0])), "a3":np.zeros((1, x.shape[0])), "z2":np.zeros((3, x.shape[0])), "a2":np.zeros((3, x.shape[0])),
             "z1":np.zeros((3, x.shape[0])), "a1":np.zeros((3, x.shape[0])), "w3":np.zeros((1, 3)), "w2":np.zeros((3, 3)), "w1":np.zeros((3, 2))}

    # training process
    acc = 0.0
    pre_acc = 0.0
    history = list()
    final = 0
    for i in range(30000):
        pred_y = forward()
        L = loss(y, pred_y)
        history.append(L[0][0])
        acc = accuracy(y, pred_y)
        if (i % 200) == 0 or acc == 1.0:
            print("---------------------")
            print(f'epoch: {i+1}, loss: {L[0][0]}, acc: {acc}')
        if acc == 1.0 or i == 30000-1:
            final = i
            break
      backward(w1, w2, w3, pred_y)

    # show record
    show_result(x, y, pred_y)
    plt.title("Learning Curve")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(range(final+1),history)

if __name__ == '__main__':
    print("==============Linear==============\n")
    linear_exp()
    print("\n\n")
    print("==============XOR==============\n")
    XOR_exp()
