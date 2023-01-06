from utils import *
from model import *
from dataloader import *

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def train(args, model, optimizer, data_type="icelvr"):
    # build iterative dataset
    if data_type == "icelvr":
        dataset = iter(sample_data(path, args.batch, args.img_size))
    else:
        dataset = iter(sample_CelebA_data(path, args.batch, args.img_size))
    # max binary number
    n_bins = 2.0 ** args.n_bits

    # for testing model
    z_sample = []
    c_sample = []
    # channel=3(R,G,B)
    # z shapes of each block
    z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
    # initialize z
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device))
    for c in z_shapes:
        c_new = torch.randperm(40)[:1]
        c_sample.append(c_new.to(device))

    # use progress bar
    with tqdm(range(args.iter)) as pbar:
        # training loop
        for i in pbar:
            # get batch
            image, label = next(dataset)
            #image = next(dataset)
            image = image.to(device)

            # image transformation
            # R,G,B = [0, 256]
            image = image * 255

            if args.n_bits < 8:
                image = torch.floor(image / 2 ** (8 - args.n_bits))

            image = image / n_bins - 0.5

            # first iteration
            # no update
            if i == 0:
                with torch.no_grad():
                    #log_p, logdet, _ = model.module(
                    #    image + torch.rand_like(image) / n_bins
                    #)
                    # output: log_p, logdet, z_out
                    log_p, logdet, _ = model(
                        image + torch.rand_like(image) / n_bins, label
                    )

                    continue

            else:
                log_p, logdet, _ = model(image + torch.rand_like(image) / n_bins, label)

            # compute logdet
            logdet = logdet.mean()

            loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
            model.zero_grad()
            # compute gradient
            loss.backward()
            # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
            warmup_lr = args.lr
            # set optimizer lr
            optimizer.param_groups[0]["lr"] = warmup_lr
            # update model weight
            optimizer.step()

            # show progress bar
            pbar.set_description(
                "Loss: {:.5f}; logP: {:.5f}; logdet: {:.5f}; lr: {:.7f}".format(loss.item(), log_p.item(), log_det.item(), warmup_lr)
            )

            # save sample image
            if i % 100 == 0:
                # testing without update
                # use sample z perform reverse flow to generate image
                with torch.no_grad():
                    utils.save_image(
                        model_single.reverse(z_sample, c_sample).cpu().data,
                        "sample/{}.png".format(str(i + 1).zfill(6)),
                        normalize=True,
                        nrow=10,
                        range=(-0.5, 0.5),
                    )

            # save model weight
            if i % 10000 == 0:
                torch.save(
                    model.state_dict(), "checkpoint/model_{}.pt".format(str(i + 1).zfill(6))
                )
                torch.save(
                    optimizer.state_dict(), "checkpoint/optim_{}.pt".format(str(i + 1).zfill(6))
                )

# test
def test(args):
    net = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    ).to(device)
    net.load_state_dict(args.ckpt_path)

    cond_data = torch.load(args.cond_data)
    original, cond_img = cond_data["original"], cond_data["cond_img"].to(device)


    # style transfer
    synth_img, target = style_transfer(net, original, cond_img, target_index=args.index)

    origin_concat = torchvision.utils.make_grid(original, nrow=3, padding=2, pad_value=255)
    img_concat = torchvision.utils.make_grid(synth_img, nrow=3, padding=2, pad_value=255)
    torchvision.utils.save_image(origin_concat, args.output_dir + 'original.png')
    torchvision.utils.save_image(img_concat, args.output_dir + 'synthesized.png')
    torchvision.utils.save_image(target, args.output_dir + 'cond_img.png')
    
    cond = cond_data["label"].to(device)
    # manipulation
    man_img, target = manipulate(net, img, cond)
    
    img_concat = torchvision.utils.make_grid(man_img, nrow=2, padding=2, pad_value=255)
    torchvision.utils.save_image(img_concat, args.output_dir + 'manipulated.png')

@torch.no_grad() # model in style_transfer() does not calculate gradient
# orginal -> cond_img interpolation
def style_transfer(net, original, cond_img, target_index=0, img_size=64):
    B = original.size(0)
    target = [cond_img[target_index] for _ in range(B)]
    target = torch.stack(target)
    log_p_sum, logdet, z = net(original, cond_img)
    reconstructed, _ = net.reverse(z, target, reconstruct=True)
    reconstructed = torch.sigmoid(reconstructed)
    return reconstructed, cond_img[target_index]

def manipulate(net, img, cond, target_index=0, img_size=64, n_step=6):
    log_p_sum, logdet, z_pos = net(img, cond)
    log_p_sum, logdet, z_neg = net(img)
    direction = z_pos - z_neg
    step = direction / n_step
    man_img = []
    z = z_neg
    for i in range(n_step):
        z += step
        new_img, _ = net.reverse(z, cond, reconstruct=True)
        man_img.append(new_img)
    return man_img

def str2bool(s):
    return s.lower().startswith('t')

if __name__ == "__main__":
    '''
        icelver
    '''
    # set parameters
    args = set_args()
    print(args)

    # construct model
    model_single = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    )
    #model = nn.DataParallel(model_single)
    model = model_single
    model = model.to(device)

    # set optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # train
    train(args, model, optimizer)

    torch.save(
        model.state_dict(), "checkpoint/model_{}.pt".format(str(1000).zfill(6))
    )
    torch.save(
        optimizer.state_dict(), "checkpoint/optim_{}.pt".format(str(1000).zfill(6))
    )

    final_net = torch.load('./checkpoint/model_200000.pkl')

    '''
        test
    '''
    test_dataset = TestingDataset('test.json')
    test_loader = torch.utils.data.DataLoader(test_dataset, 32, shuffle=False)

    for labels in test_loader:
        z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (32, 104))))
        img = netG(z, labels.to(device))
        img = F.interpolate(img, size=64)
        print('Acc = {}'.format(evaluator.eval(img, labels)))
        save_image(make_grid(img * 0.5 + 0.5), './cNF_test.png')

    new_test_dataset = TestingDataset('new_test.json')
    new_test_loader = torch.utils.data.DataLoader(new_test_dataset, 32, shuffle=False)

    for labels in new_test_loader:
        z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (32, 104))))
        img = netG(z, labels.to(device))
        img = F.interpolate(img, size=64)
        print('Acc = {}'.format(evaluator.eval(img, labels)))
        save_image(make_grid(img * 0.5 + 0.5), './cNF_new_test.png')


###########################################################################


    '''
        CelebA
    '''
    # construct model
    model_single = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    )
    #model = nn.DataParallel(model_single)
    model = model_single
    model = model.to(device)

    # set optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # train
    train(args, model, optimizer, "CelebA")

    torch.save(
        model.state_dict(), "checkpoint/model_{}.pt".format(str(1000).zfill(6))
    )
    torch.save(
        optimizer.state_dict(), "checkpoint/optim_{}.pt".format(str(1000).zfill(6))
    )

    '''
        test
    '''
    args = set_test_args()
    test(args)
