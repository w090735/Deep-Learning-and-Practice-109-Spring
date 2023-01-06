from utils import *
from dataloader imort *

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
        #loss = criterion(output, torch.max(target, 1)[1])
        train_loss += loss.item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss.backward()
        optimizer.step()
      
    acc = 100. * correct / len(train_loader.dataset)
    train_loss /= len(train_loader.dataset)
    print('Train Epoch: {} \tLoss: {:.6f} \tAccuracy: {:.1f}%'.format(
                epoch, train_loss, acc))
    return acc
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    result = np.zeros(7025)
    index = 0
    with torch.no_grad():
        for data, target in test_loader:
            # nnl_loss must use long
            data, target = data.to(device,dtype=torch.float), target.to(device,dtype=torch.long)
            output = model(data)
            # first create a CrossEntropyLoss class
            # then calculate loss
            test_loss += nn.CrossEntropyLoss()(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            for i in range(4):
                result[index] = pred[i]
                index += 1
                if index == 7025:
                    break
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))
    return acc, result

if __name__ == '__main__':
    '''
        set args
    '''
    args = set_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    #train_kwargs = {'batch_size': args.batch_size, 'shuffle': args.shuffle}
    #test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': args.test_shuffle}
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 0,
                'pin_memory': True,
                'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    '''
        train models
    '''
    path = "data/"
    train_dataset = RetinopathyLoader(path, 'train')
    test_dataset = RetinopathyLoader(path, 'test')
    train_loader = DataLoader(train_dataset, **train_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)

    acc_train = np.zeros((4, args.epochs))
    acc_test = np.zeros((4, args.epochs))

    acc_max = [0, 0, 0, 0]
    pred_best = np.zeros((4,7025))

    model_list = [models.resnet18(pretrained=False), models.resnet18(pretrained=True),
                 models.resnet50(pretrained=False), models.resnet50(pretrained=True)]

    for i in range(4):
        model = model_list[i]

        # fix params of pretrained models
        if i == 1 or i == 3:
            for param in model.parameters():
                param.requires_grad = False

        # set final output layer
        if i < 2:
            model.fc = nn.Linear(in_features=512, out_features=5, bias=True)
        else:
            model.fc = nn.Linear(in_features=2048, out_features=5, bias=True)

        if torch.cuda.is_available():
            model.cuda()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=7e-4)
        for epoch in range(1, args.epochs + 1):
            acc_train[i][epoch-1] = train(args, model, device, train_loader, optimizer, epoch)
            acc_test[i][epoch-1], pred = test(model, device, test_loader)
            if acc_test[i][epoch-1] > acc_max[i]:
                acc_max[i] = acc_test[i][epoch-1]
                pred_best[i] = pred

    '''
        accuracy figure
    '''
    model_str = ["(w/o pretraining)", "(with pretraining)"]
    plt.title("ResNet18")
    for i in range(2):
        plt.plot(range(1, args.epochs + 1), acc_train[i], label="Train"+model_str[i])
        plt.plot(range(1, args.epochs + 1), acc_test[i], label="Test"+model_str[i])
    plt.legend()
    plt.show()
    plt.title("ResNet50")
    for i in range(2,4):
        plt.plot(range(1, args.epochs + 1), acc_train[i], label="Train"+model_str[i-2])
        plt.plot(range(1, args.epochs + 1), acc_test[i], label="Test"+model_str[i-2])
    plt.legend()
    plt.show()

    '''
        max accuracy
    '''
    acc_str = ["ResNet18(w/o pretraining)","ResNet18(with pretraining)",
               "ResNet50(w/o pretraining)", "ResNet50(with pretraining)"]
    for i in range(4):
        print("{}: {}".format(acc_str[i],acc_max[i]))

    '''
        confusion matrix
    '''
    confusion = np.zeros((4, 5, 5))
    for i in range(4):
        for item in range(7025):
            confusion[i][int(label[item])][int(pred_best[i][item])] += 1

    for i in range(4):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # i: true, j: predict
        total = np.sum(confusion[i], axis=1)
        norm = np.zeros(confusion[i].shape)
        for j in range(confusion[i].shape[0]):
            norm[j] = confusion[i][j] / total[j]
        cax = ax.matshow(norm, cmap=plt.cm.Blues, interpolation='nearest')
        ax.set_title('Normalized Confusion Matrix_'+acc_str[i])
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        for i in range(2):
            for j in range(2):
                ax.text(i, j, '{:.2f}'.format(norm[j][i]), va='center', ha='center')
        fig.colorbar(cax)
        plt.show()
