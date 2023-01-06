from utils import *
from model import *
from dataloader import *

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
coco_path = "/home/auser03/.fastai/data/coco_sample/train_sample"
path = coco_path

def train_model(model, train_dl, epochs, display_every=200):
    data = next(iter(train_dl)) # getting a batch for visualizing the model output after fixed intrvals
    for e in range(epochs):
        loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to 
        i = 0                                  # log the losses of the complete network
        for data in tqdm(train_dl):
            model.setup_input(data) 
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['L'].size(0)) # function updating the log objects
            i += 1
            if i % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
                log_results(loss_meter_dict) # function to print out the losses
                visualize(model, data, save=False) # function displaying the model's outputs

def test_model(model, val_dl):
    data = next(iter(val_dl))
    loss_meter_dict = create_loss_meters()
    for data in tqdm(val_dl):
        model.setup_input(data)
        model.forward()
        update_losses(model, loss_meter_dict, count=data['L'].size(0))
    log_results(loss_meter_dict)
    visualize(model, data, save=False)

def build_res_unet(n_input=1, n_output=2, size=256):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    body = create_body(resnet18, pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G

def pretrain_generator(net_G, train_dl, opt, criterion, epochs):
    for e in range(epochs):
        loss_meter = AverageMeter()
        for data in tqdm(train_dl):
            L, ab = data['L'].to(device), data['ab'].to(device)
            preds = net_G(L)
            loss = criterion(preds, ab)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            loss_meter.update(loss.item(), L.size(0))
            
        print(f"Epoch {e + 1}/{epochs}")
        print(f"L1 Loss: {loss_meter.avg:.5f}")

if __name__ == '__main__':
    '''
        dataset
    '''
    print(coco_path)
    paths = glob.glob(path + "/*.jpg") # Grabbing all the image file names
    np.random.seed(123)
    paths_subset = np.random.choice(paths, 10_000, replace=False) # choosing 1000 images randomly
    rand_idxs = np.random.permutation(10_000)
    train_idxs = rand_idxs[:8000] # choosing the first 8000 as training set
    val_idxs = rand_idxs[8000:] # choosing last 2000 as validation set
    train_paths = paths_subset[train_idxs]
    val_paths = paths_subset[val_idxs]
    print(len(train_paths), len(val_paths))

    '''
        show dataset image
    '''
    _, axes = plt.subplots(4, 4, figsize=(10, 10))
    for ax, img_path in zip(axes.flatten(), train_paths):
        ax.imshow(Image.open(img_path))
        ax.axis("off")

    '''
        dataloader
    '''
    train_dl = make_dataloaders(paths=train_paths, split='train')
    val_dl = make_dataloaders(paths=val_paths, split='val')

    '''
        print dataset shape
    '''
    data = next(iter(train_dl))
    Ls, abs_ = data['L'], data['ab']
    print(Ls.shape, abs_.shape)
    print(len(train_dl), len(val_dl))

    '''
        print discriminator
    '''
    print(PatchDiscriminator(3))

    discriminator = PatchDiscriminator(3)
    dummy_input = torch.randn(16, 3, 256, 256) # batch_size, channels, size, size
    out = discriminator(dummy_input)
    print(out.shape)

    '''
        pretrain generator in coco dataset
    '''
    net_G = build_res_unet(n_input=1, n_output=2, size=256)
    opt = optim.Adam(net_G.parameters(), lr=1e-4)
    criterion = nn.L1Loss()        
    pretrain_generator(net_G, train_dl, opt, criterion, 20)
    torch.save(net_G.state_dict(), "res18-unet.pt")

    torch.save(model.state_dict(), "resnetG-pixelD100.pt")


    '''
        train gan
    '''
    net_G = build_res_unet(n_input=1, n_output=2, size=256)
    net_G.load_state_dict(torch.load("res18-unet.pt", map_location=device))
    model = MainModel(net_G=net_G)
    train_model(model, train_dl, 80)
    torch.save(model.state_dict(), "resnetG-patchD.pt")

    '''
        test gan
    '''
    final_model = torch.load('resnetG-patchD.pt')
    test_model(final_model, val_dl)


###################################################################


    '''
        print discriminator
    '''
    discriminator = PixelDiscriminator(3)
    dummy_input = torch.randn(16, 3, 256, 256) # batch_size, channels, size, size
    out = discriminator(dummy_input)
    print(out.shape)

    '''
        train gan with pixel discriminator
    '''
    net_G = build_res_unet(n_input=1, n_output=2, size=256)
    net_G.load_state_dict(torch.load("res18-unet.pt", map_location=device))
    model = P_MainModel(net_G=net_G)
    train_model(model, train_dl, 80)
    torch.save(model.state_dict(), "resnetG-pixelD.pt")

    '''
        test gan with pixel discriminator
    '''
    final_model = torch.load('resnetG-pixelD.pt')
    test_model(final_model, val_dl)
