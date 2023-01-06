# log(|x|)
logabs = lambda x: torch.log(torch.abs(x))
# actnorm
class ActNorm(nn.Module):
    # set properties
    def __init__(self, in_channel, logdet=True):
        super().__init__()

        # set model parameter
        # loc: bias
        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        # add buffer
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        # log determinant
        self.logdet = logdet

    # construct model
    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    # forward pass
    def forward(self, input):
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        # h*w*sum(log(|s|))
        logdet = height * width * torch.sum(log_abs)

        # y = dot(s, x) + b = dot(s, x + loc)
        if self.logdet:
            return self.scale * (input + self.loc), logdet

        else:
            return self.scale * (input + self.loc)

    # reverse function
    # y = f(x) => x = g(y)
    # x = (y - b) / s = y / s - loc
    def reverse(self, output):
        return output / self.scale - self.loc


# invertible 1x1 convolution
class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape

        # y = W*x
        out = F.conv2d(input, self.weight)
        # h*w*log(det(W))
        logdet = (
            height * width * torch.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet

    def reverse(self, output):
        # x = W.I*y
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )


# invertible 1x1 convolution with LU decomposition
class InvConv2dLU(nn.Module):
    # set properties
    def __init__(self, in_channel):
        super().__init__()

        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    # foward pass
    def forward(self, input):
        _, _, height, width = input.shape

        # compute weight
        weight = self.calc_weight()

        # y = W*x
        out = F.conv2d(input, weight)
        logdet = height * width * torch.sum(self.w_s)

        return out, logdet

    # compute weight using LU decomposition
    def calc_weight(self):
        # W = P*L*(U + diag(s))
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    # reverse function
    def reverse(self, output):
        weight = self.calc_weight()

        # x = W.I*y
        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


# model for compute mean and log_sd
class ZeroConv2d(nn.Module):
    # construct model
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        # initialize weight and bias
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    # forward pass
    def forward(self, input):
        # padding tensor with 1
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out


# affine coupling layer
class AffineCoupling(nn.Module):
    # construct model
    def __init__(self, in_channel, filter_size=512, affine=True):
        super().__init__()

        self.affine = affine

        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    # forward pass
    def forward(self, input, cond=None):
        # x_a, x_b = split(x)
        in_a, in_b = input.chunk(2, 1)

        if self.affine:
            # log_s, t = NN(x_a)
            if cond is not None:
                log_s, t = (self.net(in_a) + self.net(cond)).chunk(2, 1)
            else:
                log_s, t = self.net(in_a).chunk(2, 1)
            # s = exp(log_s)
            # s = torch.exp(log_s)
            s = F.sigmoid(log_s + 2)
            # y = dot(s, x_b) + t
            # out_a = s * in_a + t
            out_b = (in_b + t) * s

            # sum(log(|s|))
            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)

        else:
            if cond is not None:
                net_out = self.net(in_a) + self.net(cond)
            else:
                net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None

        # y = concate(y_a, y_b)
        return torch.cat([in_a, out_b], 1), logdet

    # reverse function
    def reverse(self, output, cond=None):
        # y_a, y_b = split(y)
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            # log_s, t = NN(y_a)
            if cond is not None:
                log_s, t = (self.net(out_a) + self.net(cond)).chunk(2, 1)
            else:
                log_s, t = self.net(out_a).chunk(2, 1)
            # s = exp(log_s)
            # s = torch.exp(log_s)
            s = F.sigmoid(log_s + 2)
            # x_b = (y_b - t) / s
            # in_a = (out_a - t) / s
            in_b = out_b / s - t
            # x_a = y_a

        else:
            if cond is not None:
                net_out = self.net(out_a) + self.net(cond)
            else:
                net_out = self.net(out_a)
            in_b = out_b - net_out

        # x = concate(x_a, x_b)
        return torch.cat([out_a, in_b], 1)

# flow
class Flow(nn.Module):
    # construct model
    def __init__(self, in_channel, affine=True, conv_lu=True):
        super().__init__()

        self.actnorm = ActNorm(in_channel)

        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)

        else:
            self.invconv = InvConv2d(in_channel)

        self.coupling = AffineCoupling(in_channel, affine=affine)

    # forward pass
    def forward(self, input, cond=None):
        # flow: actnorm -> invconv -> coupling
        out, logdet = self.actnorm(input)
        out, det1 = self.invconv(out)
        if cond is not None:
            out, det2 = self.coupling(out, cond)
        else:
            out, det2 = self.coupling(out)

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2

        return out, logdet

    # reverse function
    def reverse(self, output, cond=None):
        # reverse: r_actnorm <- r_invconv <- r_coupling
        if cond is not None:
            input = self.coupling.reverse(output, cond)
        else:
            input = self.coupling.reverse(output)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)

        return input


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps

#                           z
#                           ^
#                           |
# x -> squeeze -> flow -> split -> squeeze -> flow -> zL
# ^                             |
# |-----------------------------|

# multi-scale
class Block(nn.Module):
    # construct model
    def __init__(self, in_channel, n_flow, split=True, affine=True, conv_lu=True):
        super().__init__()

        squeeze_dim = in_channel * 4

        # flow sequence
        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))

        self.split = split

        # split channel to half
        if split:
            self.prior = ZeroConv2d(in_channel * 2, in_channel * 4)

        else:
            self.prior = ZeroConv2d(in_channel * 4, in_channel * 8)

    # forward pass
    def forward(self, input, cond=None):
        b_size, n_channel, height, width = input.shape
        # squeeze
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)
        
        if cond is not None:
            c_squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
            c_squeezed = c_squeezed.permute(0, 1, 3, 5, 2, 4)
            c_out = c_squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)

        logdet = 0

        # flow
        for flow in self.flows:
            if cond is not None:
                out, det = flow(out, c_out)
            else:
                out, det = flow(out)
            logdet = logdet + det

        # split
        if self.split:
            out, z_new = out.chunk(2, 1)
            mean, log_sd = self.prior(out).chunk(2, 1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)

        else:
            zero = torch.zeros_like(out)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out

        return out, logdet, log_p, z_new

    # reverse function
    def reverse(self, output, cond=None, eps=None, reconstruct=False):
        input = output

        # r_split
        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)

            else:
                input = eps

        else:
            if self.split:
                mean, log_sd = self.prior(input).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = torch.cat([output, z], 1)

            else:
                zero = torch.zeros_like(input)
                # zero = F.pad(zero, [1, 1, 1, 1], value=1)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = z

        # r_flow
        for flow in self.flows[::-1]:
            if cond is not None:
                input = flow.reverse(input, cond)
            else:
                input = flow.reverse(input)

        b_size, n_channel, height, width = input.shape

        # r_squeeze
        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_channel // 4, height * 2, width * 2
        )

        return unsqueezed


# glow
class Glow(nn.Module):
    # define blocks
    def __init__(
        self, in_channel, n_flow, n_block, affine=True, conv_lu=True
    ):
        super().__init__()

        # initial module list
        self.blocks = nn.ModuleList()
        # n_channel: input channel
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu))
            n_channel *= 2
        # last block
        self.blocks.append(Block(n_channel, n_flow, split=False, affine=affine))

    # network forward pass
    def forward(self, input, cond=None):
        log_p_sum = 0
        logdet = 0
        out = input
        z_outs = []

        for block in self.blocks:
            if cond is not None:
                out, det, log_p, z_new = block(out, cond)
            else:
                out, det, log_p, z_new = block(out)
            z_outs.append(z_new)
            logdet = logdet + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return log_p_sum, logdet, z_outs

    # reverse function
    def reverse(self, z_list, cond=None, reconstruct=False):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                if cond is not None:
                    input = block.reverse(z_list[-1], z_list[-1], cond, reconstruct=reconstruct)
                else:
                    input = block.reverse(z_list[-1], z_list[-1], reconstruct=reconstruct)

            else:
                if cond is not None:
                    input = block.reverse(input, z_list[-(i + 1)], cond, reconstruct=reconstruct)
                else:
                    input = block.reverse(input, z_list[-(i + 1)], reconstruct=reconstruct)

        return input

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
