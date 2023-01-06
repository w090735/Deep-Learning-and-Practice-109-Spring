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