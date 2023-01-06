from utils import *


def relu():
    m = nn.ReLU()
    input = torch.range(-6, 7)
    output = m(input)
    for i in range(-8, 8, 2): 
        x = list()
        for j in range(16):
            x.append(i)
        plt.plot(x, range(-8, 8), '--', color='#e0e0eb')
        plt.plot(range(-8, 8), x, '--', color='#e0e0eb')
    plt.plot(input, output)
    plt.xlim([-6.5, 6.5])
    plt.ylim([-6.5, 6.5])
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.title("ReLU activation function")

def leaky_relu():
    slope = 0.5
    m = nn.LeakyReLU(slope)
    input = torch.range(-6, 7)
    output = m(input)
    for i in range(-8, 8, 2): 
        x = list()
        for j in range(16):
            x.append(i)
        plt.plot(x, range(-8, 8), '--', color='#e0e0eb')
        plt.plot(range(-8, 8), x, '--', color='#e0e0eb')
    plt.plot(input, output)
    plt.xlim([-6.5, 6.5])
    plt.ylim([-6.5, 6.5])
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.title("LeakyReLU activation function, negative slope={:.2f}".format(slope))

def elu():
    alpha = 0.5
    m = nn.ELU(alpha)
    input = torch.range(-6, 7, 0.1)
    output = m(input)
    for i in range(-8, 8, 2): 
        x = list()
        for j in range(16):
            x.append(i)
        plt.plot(x, range(-8, 8), '--', color='#e0e0eb')
        plt.plot(range(-8, 8), x, '--', color='#e0e0eb')
    plt.plot(input, output)
    plt.xlim([-6.5, 6.5])
    plt.ylim([-6.5, 6.5])
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.title("ELU activation function, alpha={:.2f}".format(alpha))
