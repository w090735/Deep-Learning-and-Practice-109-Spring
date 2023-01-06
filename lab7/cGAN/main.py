from utils import *
from model import *
from dataloader import *

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    args = set_args()

    # construct evaluator
    evaluator = evaluation_model("./")

    # Data
    # training dataset
    train_dataset = TrainingDataset(transform=transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
    # testing dataset
    test_dataset = TestingDataset('test.json')

    # Create the dataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    # test batch_size = 32
    test_loader = torch.utils.data.DataLoader(test_dataset, 32, shuffle=False)

    # Initialize the models
    # construct generator and initialize weight
    netG = Generator(args.latent_size).to(device)
    netG.apply(weights_init)

    # construct discriminator and initialize weight
    netD = Discriminator().to(device)
    netD.apply(weights_init)

    # Initialize loss function
    # real or fake
    adversarial_loss = nn.BCELoss()
    # BCEWithLogitsLoss() = Sigmoid() + BCELoss()
    # class
    classification_loss = nn.BCEWithLogitsLoss()

    # Setup Adam optimizers
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr_D, betas=(0.5, 0.999))
    optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr_G, betas=(0.5, 0.999))

    # Fixed noise
    fixed_noise = Variable(torch.cuda.FloatTensor(
        np.random.normal(0, 1, (32, args.latent_size))))

    '''
        train
    '''
    # Training Loop
    history = {"loss_G": [], "loss_D": [], "test_acc": []}
    iter = 0
    highest_test_acc = 0

    for epoch in range(1, args.num_epochs + 1):
        # display progress bar
        pbar = tqdm(total=len(train_loader), unit=' batches',  ascii=True)
        pbar.set_description("({}/{})".format(epoch, args.num_epochs))

        for batch_idx, (real_img, condition) in enumerate(train_loader):
            # get batch
            real_img, condition = real_img.to(device), condition.to(device)
            b_size = real_img.size(0)
            iter += 1

            # Adversarial ground truths
            real = torch.full((b_size, 1), 1., device=device)
            fake = torch.full((b_size, 1), 0., device=device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Loss for real images
            dis_output, aux_output = netD(real_img)
            D_x = dis_output.mean().item()
            # adv_loss + class_loss * weight
            d_real_loss = adversarial_loss(
                dis_output, real) + classification_loss(aux_output, condition) * args.d_aux_weight
            d_real_loss.backward()

            # Loss for fake images
            # z: N(0, 1)
            z = Variable(torch.cuda.FloatTensor(
                np.random.normal(0, 1, (b_size, args.latent_size))))
            fake_img = netG(z, condition)
            dis_output, _ = netD(fake_img.detach())
            D_G_before = dis_output.mean().item()
            d_fake_loss = adversarial_loss(dis_output, fake)
            d_fake_loss.backward()

            # Net Loss for the discriminator
            D_loss = d_real_loss + d_fake_loss
            history['loss_D'].append(D_loss.item())
            # Update parameters
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Classify all fake batch with D
            # z: N(0, 1)
            z = Variable(torch.cuda.FloatTensor(
                np.random.normal(0, 1, (b_size, args.latent_size))))
            gen_img = netG(z, condition)
            dis_output, aux_output = netD(gen_img)
            D_G_after = dis_output.mean().item()
            # adv_loss + class_loss + weight
            G_loss = adversarial_loss(
                dis_output, real) + classification_loss(aux_output, condition) * g_aux_weight(iter)
            history['loss_G'].append(G_loss.item())
            # Calculate gradients
            G_loss.backward()
            # Update parameters
            optimizer_G.step()

            # ------------
            #  Evaluation
            # ------------
            test_acc = 0
            for test_label in test_loader:
                test_label = test_label.to(device)

                with torch.no_grad():
                    test_img = netG(fixed_noise, test_label)
                    # resize image to 64x64
                    test_img = F.interpolate(test_img, size=64)
                    test_acc += evaluator.eval(test_img, test_label)

                if iter % 100 == 0:
                    # denormalize image
                    save_image(make_grid(test_img * 0.5 + 0.5),
                               './results/iter_{}.png'.format(iter))

            test_acc = test_acc / len(test_loader)
            history['test_acc'].append(test_acc)

            pbar.set_postfix({
                'D_x': D_x,
                'D_G_before': D_G_before,
                'D_G_after': D_G_after,
                'test_acc': history['test_acc'][-1]
            })
            pbar.update()

            if history['test_acc'][-1] > highest_test_acc:
                highest_test_acc = history['test_acc'][-1]
                torch.save(netG, './models/netG.pkl')
                torch.save(netD, './models/netD.pkl')

        pbar.close()

    # save training history
    with open('history.pkl', "wb") as fp:
        pickle.dump(history, fp)

    '''
        test
    '''
    final_netG = torch.load('./models/netG.pkl')

    for labels in test_loader:
        z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (32, 104))))
        img = netG(z, labels.to(device))
        img = F.interpolate(img, size=64)
        print('Acc = {}'.format(evaluator.eval(img, labels)))
        save_image(make_grid(img * 0.5 + 0.5), './ACGAN_test.png')

    new_test_dataset = TestingDataset('new_test.json')
    new_test_loader = torch.utils.data.DataLoader(test_dataset, 32, shuffle=False)

    for labels in new_test_loader:
        z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (32, 104))))
        img = netG(z, labels.to(device))
        img = F.interpolate(img, size=64)
        print('Acc = {}'.format(evaluator.eval(img, labels)))
        save_image(make_grid(img * 0.5 + 0.5), './ACGAN_new_test.png')
