# Generator
class Generator(nn.Module):
    # use transpose convolution to upsampling
    def __init__(self, latent_size):
        super(Generator, self).__init__()
        # concate latent and label
        self.input_size = latent_size + 24

        # first linear layer
        self.fc1 = nn.Linear(self.input_size, 768)
        # Transposed Convolution 2
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(768, 384, 5, 2, 0, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(True),
        )
        # Transposed Convolution 3
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(384, 256, 5, 2, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 192, 5, 2, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        # Transposed Convolution 5
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, 5, 2, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        # Transposed Convolution 6
        self.tconv6 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 8, 2, 0, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, condition):
        # z: (batch_size, latent_size)
        # condition: (batch_size, 24)
        # gen_input: (batch_size, latent_size + 24)
        gen_input = torch.cat((z, condition), -1)

        # fc1: (batch_size, 768)
        fc1 = self.fc1(gen_input)
        # fc1: (batch_size, 768, 1, 1)
        fc1 = fc1.view(-1, 768, 1, 1)
        # tconv2: (batch_size, 384, 5, 5)
        tconv2 = self.tconv2(fc1)
        # tconv3: (batch_size, 256, 17, 17)
        tconv3 = self.tconv3(tconv2)
        # tconv4: (batch_size, 192, 42, 42)
        tconv4 = self.tconv4(tconv3)
        # tconv5: (batch_size, 64, 91, 91)
        tconv5 = self.tconv5(tconv4)
        # output: (batch_size, 3, 128, 128)
        output = self.tconv6(tconv5)

        return output


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25, inplace=False),
        )
        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25, inplace=False),
        )
        # Convolution 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25, inplace=False),
        )
        # Convolution 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25, inplace=False),
        )
        # Convolution 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25, inplace=False),
        )
        # Convolution 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25, inplace=False),
        )
        # discriminator fc
        self.fc_dis = nn.Linear(13*13*512, 1)
        # aux-classifier fc
        self.fc_aux = nn.Linear(13*13*512, 24)
        # sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        # img: (batch_size, 3, 128, 128)
        batch_size = img.size(0)

        # conv1: (batch_size, 16, 64, 64)
        conv1 = self.conv1(img)
        # conv2: (batch_size, 32, 62, 62)
        conv2 = self.conv2(conv1)
        # conv3: (batch_size, 64, 31, 31)
        conv3 = self.conv3(conv2)
        # conv4: (batch_size, 128, 29, 29)
        conv4 = self.conv4(conv3)
        # conv5: (batch_size, 256, 14, 14)
        conv5 = self.conv5(conv4)
        # conv6: (batch_size, 512, 13, 13)
        conv6 = self.conv6(conv5)

        # flat6: (batch_size, 13*13*512)
        flat6 = conv6.view(batch_size, -1)
        # fc_dis: (batch_size, 1)
        fc_dis = self.fc_dis(flat6)
        # validity: (batch-size, 1)
        # real or fake
        validity = self.sigmoid(fc_dis)
        # label: (batch_size, 24)
        # class
        label = self.fc_aux(flat6)

        return validity, label

# evaluator
class evaluation_model():
    # construct model and evaluate
    def __init__(self, root):
        # load model weight
        checkpoint = torch.load(root+'classifier_weight.pth')
        # construct resnet18
        self.resnet18 = models.resnet18(pretrained=False)
        # modify last layer output channel to 24 class
        self.resnet18.fc = nn.Sequential(
            nn.Linear(512,24),
            nn.Sigmoid()
        )
        # set model weight
        self.resnet18.load_state_dict(checkpoint['model'])
        # load model to device
        self.resnet18 = self.resnet18.cuda()
        # set model to evaluate mode
        self.resnet18.eval()
        # set class number
        self.classnum = 24
    # compute acc score
    def compute_acc(self, out, onehot_labels):
        # out: (batch_size, 24)
        # onehot_labels: (batch_size, 24)
        batch_size = out.size(0)
        acc = 0
        total = 0
        for i in range(batch_size):
            # number of objects in one image
            k = int(onehot_labels[i].sum().item())
            # number of total generate class
            total += k
            # outv: top k value of fake image
            # outi: top k index of fake image
            outv, outi = out[i].topk(k)
            # lv: top k value of ground truth
            # li: top k index of ground truth
            lv, li = onehot_labels[i].topk(k)
            for j in outi:
                if j in li:
                    # predict class == ground truth
                    acc += 1
        return acc / total
    # evaluate with model
    def eval(self, images, labels):
        with torch.no_grad():
            #your image shape should be (batch, 3, 64, 64)
            # only R,G,B channel
            # classify image
            out = self.resnet18(images)
            # compute score
            acc = self.compute_acc(out.cpu(), labels.cpu())
            return acc