from utils import *
from model import *
from activation import *

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # nnl_loss must use long
        data, target = data.to(device,dtype=torch.float), target.to(device,dtype=torch.long)
        optimizer.zero_grad()
        output = model(data)
        # first create a CrossEntropyLoss class
        # then calculate loss
        loss = nn.CrossEntropyLoss()(output, target)
        train_loss += loss.item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss.backward()
        optimizer.step()
      
    acc = 100. * correct / len(test_loader.dataset)
    train_loss /= len(train_loader.dataset)
    print('Train Epoch: {} \tLoss: {:.6f} \tAccuracy: {:.1f}%'.format(
                epoch, train_loss, acc))
    return acc


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # nnl_loss must use long
            data, target = data.to(device,dtype=torch.float), target.to(device,dtype=torch.long)
            output = model(data)
            # first create a CrossEntropyLoss class
            # then calculate loss
            test_loss += nn.CrossEntropyLoss()(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))
    return acc


if __name__ == '__main__':
    '''
        set arguments
    '''
    args = set_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 4,
                        'pin_memory': True,
                        'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    '''
        train models
    '''
    train_x, train_y, test_x, test_y = read_bci_data()

    tensor_train_x = torch.Tensor(train_x)
    tensor_train_y = torch.Tensor(train_y)
    tensor_test_x = torch.Tensor(test_x)
    tensor_test_y = torch.Tensor(test_y)

    train_dataset = TensorDataset(tensor_train_x, tensor_train_y)
    test_dataset = TensorDataset(tensor_test_x, tensor_test_y)

    train_loader = DataLoader(train_dataset, **train_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)

    activates = [nn.ELU(alpha=0.8), nn.ReLU(), nn.LeakyReLU(negative_slope=0.03)]
    acc_train = np.zeros((2, 3, args.epochs))
    acc_test = np.zeros((2, 3, args.epochs))
    for net in range(2):
        for act in range(3):
            if net == 0:
                model = EEGNet(activates[act]).to(device)
            else:
                model = DeepConvNet(activates[act]).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)

            for epoch in range(1, args.epochs + 1):
                acc_train[net][act][epoch-1] = train(args, model, device, train_loader, optimizer, epoch)
                acc_test[net][act][epoch-1] = test(model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    '''
        accuracy figure
    '''
    activate_str = ["ELU","ReLU","LeakyReLU"]
    for net in range(2):
        if net == 0:
            plt.title("EEGNet")
        else:
            plt.title("DeepConvNet")
        for i in range(3):
            plt.plot(range(1, args.epochs + 1), acc_train[net][i], label=activate_str[i]+"_train")
            plt.plot(range(1, args.epochs + 1), acc_test[net][i], label=activate_str[i]+"test")
        plt.legend()
        plt.show()

    '''
        max accuracy
    '''
    max_acc = list()
    for net in range(2):
        for i in range(3):
            max_acc.append(max(acc_test[net][i]))
    print("Maximum Accuracy:")
    print("\t\tELU\tReLU\tLeakyReLU")
    print('EGGNet\t\t{:.2f}\t{:.2f}\t{:.2f}'.format(max_acc[0], max_acc[1], max_acc[2]))
    print('DeepConvNet\t{:.2f}\t{:.2f}\t{:.2f}'.format(max_acc[3], max_acc[4], max_acc[5]))

    '''
        activation function
    '''
    relu()
    leaky_relu()
    elu()
