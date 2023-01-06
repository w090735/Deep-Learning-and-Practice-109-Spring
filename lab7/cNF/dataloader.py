'''
    icelvr
'''

# retreive data from json file to numpy array
def getData(mode, filename=None):
    # training data
    if mode == 'train':
        # load json file
        with open('train.json') as f:
            train_data = json.load(f)

        # json format: {key: value} => {image_name: object_list}
        img_names = np.array(list(train_data.keys()))
        # objects: (#images, #objects)
        # nested list convert to numpy array => dtype=object
        objects = np.array(list(train_data.values()), dtype=object)

        return img_names, objects
    # testing data
    else:
        # load json file
        with open(filename) as f:
            test_data = json.load(f)

        # objects: (#images, #objects)
        # nested list convert to numpy array => dtype=object
        return np.array(test_data, dtype=object)


# training dataset
class TrainingDataset(Dataset):
    # set properties
    def __init__(self, transform):
        # path of image files
        self.img_root_path = path + "images/"
        # numpy array of image names and their corresponding objects
        self.img_names, self.objects = getData('train')
        # dataset transformation
        self.transform = transform

        # load json file of object names
        with open('objects.json') as f:
            self.objects_json = json.load(f)

    # return length of dataset
    def __len__(self):
        return len(self.objects)

    # get item of dataset[index]
    def __getitem__(self, index):
        # 24 classes, [0, 0,..., 0]
        label = [0 for i in range(24)]
        # set the one-hot encode of objects from each images
        for object_name in self.objects[index]:
            label[self.objects_json[object_name]] = 1
        # convert label from int list to float tensor
        label = torch.tensor(label, dtype=torch.float)

        # image file name with path
        path = self.img_root_path + self.img_names[index]
        # read image with R,G,B,A
        img = Image.open(path)
        # convert image to R,G,B
        img = img.convert("RGB")
        # image transformation
        img = self.transform(img)

        return img, label


# testing dataset
class TestingDataset(Dataset):
    # set property
    def __init__(self, filename):
        # numpy array of objects of each testing data
        self.objects = getData('test', filename)

        # load json file of object names
        with open('objects.json') as f:
            self.objects_json = json.load(f)

    # return length of dataset
    def __len__(self):
        return len(self.objects)

    # get item of dataset[index]
    def __getitem__(self, index):
        # 24 classes, [0, 0,..., 0]
        label = [0 for i in range(24)]
        # set the one-hot encode of objects from each images
        for object_name in self.objects[index]:
            label[self.objects_json[object_name]] = 1
        # convert label from int list to float tensor
        label = torch.tensor(label, dtype=torch.float)

        return label

def sample_data(path, batch_size, image_size, train=True):

    #dataset = datasets.ImageFolder(path, transform=transform)
    if train:
        dataset = TrainingDataset(transform=transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))
    else:
        dataset = TestingDataset('test.json')
    
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=4
            )
            loader = iter(loader)
            yield next(loader)

'''
    celebA
'''

# get image and lable
def get_CelebA_data(root_folder, subset):
    # construct image list
    img_list = os.listdir(os.path.join(root_folder, 'CelebA-HQ-img'))
    label_list = []
    f = open(os.path.join(root_folder, 'CelebA-HQ-attribute-anno.txt'), 'r')
    # first line: image number
    num_imgs = int(f.readline()[:-1])
    if subset is not None:
        num_imgs = subset
        img_list = img_list[:subset]
        print("use subset")
    # second line: 40 image attribute
    attrs = f.readline()[:-1].split(' ')
    for idx in range(num_imgs):
        # retreive each line
        line = f.readline()[:-1].split(' ')
        # line[0]: image name
        # line[2:]: image attributes
        label = line[2:]
        label = list(map(int, label))
        label_list.append(label)
    f.close()
    return img_list, label_list


# dataset
class CelebALoader(data.Dataset):
    # set properties
    def __init__(self, root_folder, trans=None, cond=False, subset=None):
        # set path
        self.root_folder = root_folder
        assert os.path.isdir(self.root_folder), '{} is not a valid directory'.format(self.root_folder)
        
        # whether use condition
        self.cond = cond
        # get image and label
        self.img_list, self.label_list = get_CelebA_data(self.root_folder, subset)
        # number of attributes
        self.num_classes = 40
        print("> Found %d images..." % (len(self.img_list)))
        # image transformation
        self.transform = trans

    # get length of dataset
    def __len__(self):
        return len(self.img_list)


    # get item of dataset[index]
    def __getitem__(self, index):
        # image file name with path
        path = self.root_folder + "CelebA-HQ-img/"+ self.img_list[index]
        # read image with R,G,B,A
        img = Image.open(path)
        # image transformation
        img = self.transform(img)
        
        if self.cond:
            # convert label from int list to float tensor
            label = torch.tensor(self.label_list[index], dtype=torch.float)
            
            return img, label
        else:
            return img

def sample_CelebA_data(path, batch_size, image_size):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    #dataset = datasets.ImageFolder(path, transform=transform)
    dataset = CelebALoader(path, trans=transform, cond=True)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=4
            )
            loader = iter(loader)
            yield next(loader)