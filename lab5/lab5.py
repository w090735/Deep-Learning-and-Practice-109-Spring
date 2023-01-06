from utils import *
from dataloader import *
from model import *

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def train(vae,loader_train,optimizer,teacher_forcing_ratio,kl_weight,tensor2string):
    """train 1 epoch
    :param vae: model
    :param loader_train: loader_train
    :param optimizer: sgd optimizer
    :param tensor2string: function(tensor){ return string }  (cutoff EOS automatically)
    :returns: CEloss, KLloss, BLEUscore
    """
    vae.train()
    total_CEloss=0
    total_KLloss=0
    total_BLEUscore=0
    for word_tensor,tense_tensor in loader_train:
        optimizer.zero_grad()
        word_tensor,tense_tensor=word_tensor[0],tense_tensor[0]
        word_tensor,tense_tensor=word_tensor.to(device),tense_tensor.to(device)

        # init hidden_state
        h0 = vae.encoder.init_h0(vae.hidden_size - vae.conditional_size)
        c = vae.tense_embedding(tense_tensor).view(1, 1, -1)
        encoder_hidden_state = torch.cat((h0, c), dim=-1)
        # init cell_state
        encoder_cell_state = vae.encoder.init_c0()

        # forwarding one word by calling VAE Model forwarding
        """set teacher forcing ratio = 
        """
        use_teacher_forcing=True if random.random()<teacher_forcing_ratio else False
        predict_output,predict_distribution,mean,logvar=vae(word_tensor,encoder_hidden_state,encoder_cell_state,c,use_teacher_forcing)
        CEloss,KLloss = loss_function(predict_distribution,predict_output.size(0),word_tensor.view(-1),mean,logvar)
        loss = CEloss + kl_weight * KLloss
        total_CEloss+=CEloss.item()
        total_KLloss+=KLloss.item()
        predict,target=tensor2string(predict_output),tensor2string(word_tensor)
        total_BLEUscore+=compute_bleu(predict,target)

        #update
        loss.backward()
        optimizer.step()

    return total_CEloss/len(loader_train), total_KLloss/len(loader_train), total_BLEUscore/len(loader_train)


def evaluate(vae,loader_test,tensor2string):
    """
    :param tensor2string: function(tensor){ return string }  (cutoff EOS automatically)
    :return: [[input,target,predict],[input,target,predict]...], BLEUscore
    """
    vae.eval()
    re=[]
    total_BLEUscore=0
    with torch.no_grad():
        for in_word_tensor,in_tense_tensor,tar_word_tensor,tar_tense_tensor in loader_test:
            in_word_tensor,in_tense_tensor=in_word_tensor[0].to(device),in_tense_tensor[0].to(device)
            tar_word_tensor,tar_tense_tensor=tar_word_tensor[0].to(device),tar_tense_tensor[0].to(device)

            # init hidden_state
            h0 = vae.encoder.init_h0(vae.hidden_size - vae.conditional_size)
            in_c = vae.tense_embedding(in_tense_tensor).view(1, 1, -1)
            encoder_hidden_state = torch.cat((h0, in_c), dim=-1)
            # init cell_state
            encoder_cell_state = vae.encoder.init_c0()

            # forwarding one word by calling VAE Mode inference
            tar_c=vae.tense_embedding(tar_tense_tensor).view(1,1,-1)
            predict_output=vae.inference(in_word_tensor,encoder_hidden_state,encoder_cell_state,tar_c)
            target_word=tensor2string(tar_word_tensor)
            predict_word=tensor2string(predict_output)
            re.append([tensor2string(in_word_tensor),target_word,predict_word])
            total_BLEUscore+=compute_bleu(predict_word,target_word)

    return re, total_BLEUscore/len(loader_test)

# generate 100 word with 4 tenses
def generateWord(vae,latent_size,tensor2string):
    vae.eval()
    re = []
    with torch.no_grad():
        for i in range(100):
            # latent: (1, 1, latent_size) of points sample from N(0, 1)
            latent = torch.randn(1, 1, latent_size).to(device)
            tmp = []
            for tense in range(4):
                # generate: predict word index tensor
                word = tensor2string(vae.generate(latent, tense))
                tmp.append(word)

            re.append(tmp)
    return re

if __name__ == '__main__':
    args = set_args()
    tf_list=[]
    # epochs=500 args.epochs

    '''
        monotonic graph
    '''
    kl_weight_list=[]
    # kl_annealing_type='monotonic' args.kl_annealing_type
    # time=2 args.time

    for epoch in range(1, args.epochs+1):
        kl_weight_list.append(get_kl_weight(epoch, args.epochs, args.kl_annealing_type, args.time))

    plt.plot(range(1, args.epochs+1), kl_weight_list, linestyle=':')
    plt.show()

    '''
        train VAE
    '''
    # dataloader
    dataset_train = MyDataSet(path='train.txt', is_train=True)
    loader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
    dataset_test = MyDataSet(path='test.txt', is_train=False)
    loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
    print(f'MAX_LENGTH: {dataset_train.max_length}')

    # VAE model
    vae = VAE(args.input_size, args.hidden_size, args.latent_size, args.conditional_size, max_length=dataset_train.max_length).to(device)

    optimizer = optim.SGD(vae.parameters(), lr=args.LR)
    CEloss_list, KLloss_list, BLEUscore_list, teacher_forcing_ratio_list, kl_weight_list=[],[],[],[],[]
    best_BLEUscore = 0
    best_model_wts = None
    for epoch in range(1, args.epochs+1):
        """
        train
        """
        # get teacher_forcing_ratio & kl_weight
        teacher_forcing_ratio = get_teacher_forcing_ratio(epoch, args.epochs)
        kl_weight = get_kl_weight(epoch, args.epochs, args.kl_annealing_type, args.time)
        CEloss, KLloss, _ = train(vae, loader_train, optimizer, teacher_forcing_ratio, kl_weight, dataset_train.tensor2string)
        CEloss_list.append(CEloss)
        KLloss_list.append(KLloss)
        teacher_forcing_ratio_list.append(teacher_forcing_ratio)
        kl_weight_list.append(kl_weight)
        print(f'epoch{epoch:>2d}/{epochs}  tf_ratio:{teacher_forcing_ratio:.2f}  kl_weight:{kl_weight:.2f}')
        print(f'CE:{CEloss:.4f} + KL:{KLloss:.4f} = {CEloss+KLloss:.4f}')

        """
        validate
        """
        conversion, BLEUscore = evaluate(vae, loader_test, dataset_test.tensor2string)
        # generate words
        #generated_words=generateWord(vae,latent_size,dataset_test.tensor2string)
        #Gaussianscore=get_gaussian_score(generated_words)
        BLEUscore_list.append(BLEUscore)
        print(conversion)
        #print(generated_words)
        print(f'BLEU socre:{BLEUscore:.4f}') # Gaussian score:{Gaussianscore:.4f}')
        print()

        """
        update best model wts
        """
        if BLEUscore > best_BLEUscore:
            best_BLEUscore = BLEUscore
            best_model_wts = copy.deepcopy(vae.state_dict())
            # save model
            torch.save(best_model_wts,os.path.join('models',f'{args.kl_annealing_type}_time{args.time}_epochs{args.epochs}.pt'))
            fig = plot(epoch,CEloss_list,KLloss_list,BLEUscore_list,teacher_forcing_ratio_list,kl_weight_list)
            fig.savefig(os.path.join('results',f'{args.kl_annealing_type}_time{args.time}_epochs{args.epochs}.png'))
            
    torch.save(best_model_wts,os.path.join('models',f'{args.kl_annealing_type}_time{args.time}_epochs{args.epochs}.pt'))
    fig = plot(epochs,CEloss_list,KLloss_list,BLEUscore_list,teacher_forcing_ratio_list,kl_weight_list)
    fig.savefig(os.path.join('results',f'{args.kl_annealing_type}_time{args.time}_epochs{args.epochs}.png'))

    '''
        test VAE with BLEU score and Gaussian score
    '''
    # dataloader
    dataset_test = MyDataSet(path='test.txt', is_train=False)
    loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    # VAE model
    vae = VAE(args.input_size, args.hidden_size, args.latent_size, args.conditional_size, args.max_length).to(device)
    vae.load_state_dict(torch.load(os.path.join('models',args.file_path)))

    """
    test
    """
    total_BLEUscore = 0
    total_Gaussianscore = 0
    # test_time = 20 args.test_time
    for i in range(args.test_time):
        conversion, BLEUscore = evaluate(vae, loader_test, dataset_test.tensor2string)
        # generate words
        generated_words = generateWord(vae, args.latent_size, dataset_test.tensor2string)
        Gaussianscore = get_gaussian_score(generated_words)
        print('test.txt prediction:')
        print(conversion)
        print('generate 100 words with 4 different tenses:')
        print(generated_words)
        print(f'BLEU socre:{BLEUscore:.2f}')
        print(f'Gaussian score:{Gaussianscore:.2f}')
        total_BLEUscore += BLEUscore
        total_Gaussianscore += Gaussianscore
    print()
    print(f'avg BLEUscore {total_BLEUscore/args.test_time:.2f}')
    print(f'avg Gaussianscore {total_Gaussianscore/args.test_time:.2f}')
