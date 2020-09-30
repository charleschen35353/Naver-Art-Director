import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.autograd import Variable
from PIL import Image, ImageOps
import model_all
from nsml import DATASET_PATH
from creative_dataset import CreativeDataset
from torchvision import datasets, transforms


# parser = get_parser()
use_gpu = torch.cuda.is_available()

class TenCrop(object):
    def __init__(self, size, normalize=None):
        self.size = size
        self.normalize = normalize

    def __call__(self, img):
        img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
        center_crop = transforms.CenterCrop(self.size)
        resize = transforms.Resize((self.size,self.size))
        img_list = []
        w, h = img.size
        for image in [img, img_flip]:
            img_list.append(resize(image))
            img_list.append(center_crop(image))
            img_list.append(image.crop((0, 0, self.size, self.size)))
            img_list.append(image.crop((w-self.size, 0, w, self.size)))
            img_list.append(image.crop((0, h - self.size, self.size, h)))
            img_list.append(image.crop((w-self.size, h-self.size, w, h)))
        imgs = None
        to_tensor = transforms.ToTensor()
        for image in img_list:
            if imgs is None:
                temp_img = to_tensor(image)
                imgs = self.normalize(temp_img)
            else:
                temp_img = to_tensor(image)
                temp_img = self.normalize(temp_img)
                imgs = torch.cat((imgs, temp_img))
        return imgs

# for vincent-cu002
# image_datasets = {x: CreativeDataset(os.path.join(DATASET_PATH, 'train', x) + '/%s_data' % x,
#                                      os.path.join(DATASET_PATH, 'train'), data_transforms[x])
#                   for x in ['train', 'test']}

# for vincent-cu012

print("path of dataset in nsml: ")

for root, dirs, files in os.walk(DATASET_PATH):
    path = root.split(os.sep)
    print((len(path) - 1) * '---', os.path.basename(root))
    for file in files:
        if not file.split('.')[-1] == 'jpg':
            print(len(path) * '---', file)



# image_datasets = {x: CreativeDataset(os.path.join(DATASET_PATH, x) + '/%s_data' % x,
#                                      os.path.join(DATASET_PATH), data_transforms[x])
#                   for x in ['train', 'test']}

#class_names = image_datasets['train'].classes


def get_transforms(args):
    #train_tf = [
    #        transforms.Resize((255, 255)),
    #        transforms.RandomCrop((224, 224))
    #        ]
    #transforms.RandomResizedCrop(224),
    #transforms.RandomRotation(15),
    #print(args.tmp)
    #exec(args.tmp)
    if args.train_tf != 'Preprocessed':
        train_tf = eval(args.train_tf)
        train_tf = train_tf + [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
    test_tf = eval(args.test_tf)
    #test_tf = [
    #        transforms.Resize((resize, resize)),
    #        transforms.CenterCrop((crop, crop))
    #        ]

    if args.use_multicrop:
        test_tf = test_tf + [
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack([transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops]))
                ]
    else:
        test_tf = test_tf + [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]

    data_transforms = {
        'train': transforms.Compose(train_tf) if args.train_tf != 'Preprocessed' else args.train_tf,
        'test': transforms.Compose(test_tf)
    }

    print(data_transforms)
    #import ipdb
    #ipdb.set_trace()
    return data_transforms

def data_loader_k_fold(args, train=True, batch_size=50):
    data_transforms = get_transforms(args)
    train_dataset = CreativeDataset(
        os.path.join(DATASET_PATH, 'train', 'train') + '/%s_data' % 'train',
        os.path.join(DATASET_PATH, 'train'),
        args,
        data_transforms['train'],
        [args.k_fold, args.k_fold_order, 'train']
    )
    valid_dataset = CreativeDataset(
        os.path.join(DATASET_PATH, 'train', 'train') + '/%s_data' % 'train',
        os.path.join(DATASET_PATH, 'train'),
        args,
        data_transforms['train'],
        [args.k_fold, args.k_fold_order, 'valid']
    )

    image_datasets = {x: CreativeDataset(os.path.join(DATASET_PATH, 'train', x) + '/%s_data' % x,
                                    os.path.join(DATASET_PATH, 'train'),args, data_transforms[x])
                  for x in ['test']}
    image_datasets['train'] = train_dataset
    image_datasets['valid'] = valid_dataset

    dataset_sizes = {x: len(image_datasets[x]) for x in ['test', 'train', 'valid']}

    print("train data size : ", dataset_sizes['train'])
    print("valid data size : ", dataset_sizes['valid'])
    print("test data size : ", dataset_sizes['test'])
    if train:
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                      batch_size=batch_size,
                                                      pin_memory=True,
                                                      shuffle=True,
                                                      num_workers=args.num_workers)
                       for x in ['test', 'train', 'valid']}
        return dataloaders, dataset_sizes
        
    else:
        test_dl = torch.utils.data.DataLoader(image_datasets['test'],
                                              batch_size=batch_size,
                                              pin_memory=True,
                                              num_workers=args.num_workers,
                                              shuffle=True)
        return test_dl

def data_loader(args, train=True, batch_size=50):
    if args.k_fold > 0:
        return data_loader_k_fold(args, train, batch_size)
    
    data_transforms = get_transforms(args)
    image_datasets = {x: CreativeDataset(os.path.join(DATASET_PATH, 'train', x) + '/%s_data' % x,
                                     os.path.join(DATASET_PATH, 'train'), args, data_transforms[x])
                  for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    print("train data size : ", dataset_sizes['train'])
    print("test data size : ", dataset_sizes['test'])
    if train:
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      pin_memory=True,
                                                      num_workers=args.num_workers)
                       for x in ['train', 'test']}

        return dataloaders, dataset_sizes
    else:
        test_dl = torch.utils.data.DataLoader(image_datasets['test'],
                                              batch_size=batch_size,
                                              pin_memory=True,
                                              num_workers=args.num_workers,
                                              shuffle=True)
        '''
        data = []
        for i, d in enumerate(test_dl):
            inputs, labels, ctrs, sexs, age_areas, adids = d
            data.append((inputs, labels, ctrs, sexs, age_areas))
        return data
        '''

        return test_dl

def infer_regression(input, model):
    # load data into torch tensor
    model.eval()
    # get the inputs
    inputs, labels, ctrs, sexs, age_areas = None, None, None, None, None
    for item in input:
        i, l, c, s, a = item
        if inputs is None:
            inputs, labels, ctrs, sexs, age_areas = i, l, c, s, a
        else:
            inputs = torch.cat((inputs, i), 0)
            labels = torch.cat((labels, l), 0)
            ctrs = torch.cat((ctrs, c), 0)
            sexs = torch.cat((sexs, s), 0)
            age_areas = torch.cat((age_areas, a), 0)

    labels = labels.view(labels.size(0), -1)
    ctrs = ctrs.view(ctrs.size(0), -1)
    sexs = sexs.view(sexs.size(0), -1)

    # wrap them in Variable
    if use_gpu:
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        ctrs = Variable(ctrs.float().cuda())
        sexs = Variable(sexs.float().cuda())
        age_areas = Variable(age_areas.float().cuda())
    else:
        inputs, labels = Variable(inputs), Variable(labels)
        ctrs, sexs, age_areas = Variable(ctrs.float()), Variable(sexs.float()), Variable(age_areas.float())

    clean_state = model(inputs, sexs, age_areas)
    return list(zip(list(clean_state.data.cpu().squeeze().tolist()),
                    list(ctrs.data.cpu().squeeze().tolist())))

def infer_classification(input, model, top_k=1):
    # load data into torch tensor
    model.eval()
    # get the inputs
    inputs, labels, ctrs, sexs, age_areas = None, None, None, None, None
    for item in input:
        i, l, c, s, a = item
        if inputs is None:
            inputs, labels, ctrs, sexs, age_areas = i, l, c, s, a
        else:
            inputs = torch.cat((inputs, i), 0)
            labels = torch.cat((labels, l), 0)
            ctrs = torch.cat((ctrs, c), 0)
            sexs = torch.cat((sexs, s), 0)
            age_areas = torch.cat((age_areas, a), 0)

    labels = labels.view(labels.size(0), -1)
    ctrs = ctrs.view(ctrs.size(0), -1)
    sexs = sexs.view(sexs.size(0), -1)

    # wrap them in Variable
    if use_gpu:
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        ctrs = Variable(ctrs.float().cuda())
        sexs = Variable(sexs.float().cuda())
        age_areas = Variable(age_areas.float().cuda())
    else:
        inputs, labels = Variable(inputs), Variable(labels)
        ctrs, sexs, age_areas = Variable(ctrs.float()), Variable(sexs.float()), Variable(age_areas.float())

    outputs = model(inputs, sexs, age_areas)
    _, preds = torch.max(outputs.data, 1)

    clean_state = model(inputs, sexs, age_areas)
    batch_size, all_cls = clean_state.size()
    # prediction format: ([torch.Tensor], [toch.Tensor)
    prediction = F.softmax(clean_state).topk(min(top_k, all_cls))

    # output format
    # [[(key, prob), (key, prob)... ], ...]

    return prediction[0].data.cpu().squeeze().tolist() ,prediction[1].data.cpu().squeeze().tolist()
    #return list(zip(list(prediction[0].data.cpu().squeeze().tolist()),
    #                list(prediction[1].data.cpu().squeeze().tolist())))
'''
def get_infer_results(model,args):
    def _batch_loader(iterable, n=1):
        length = len(iterable)
        for n_idx in range(0, length, n):
            yield iterable[n_idx:min(n_idx + n, length)]

    print('[infer] Start prediction for test data')
    input_data = data_loader(train=False,batch_size=1)
    option = args.methods
    full_pred = []
    print('Start prediction')

    for data in _batch_loader(input_data, batch_size):
        print("data:", data)
        if options == 'regression':
            res = infer_regression(data, model)
        else:  # classification
            res = infer_classification(data, model, top_k=1)
        print("res:", res)
        full_pred += res

    print('Inference done')
    return full_pred
'''
def get_infer_results(model,args):
    print('[infer] Start prediction for test data')
    test_dl = data_loader(train=False,batch_size=1)
    options= args.method
    for i, d in enumerate(test_dl):

        data=[]
        inputs, labels, ctrs, sexs, age_areas, adids = d
        data.append((inputs, labels, ctrs, sexs, age_areas))
        if options == 'regression':
            predict_val,real_ctr = infer_regression(data, model)
        else:  # classification
            predict_val,real_ctr = infer_classification(data, model, top_k=1)
        #import ipdb
        #ipdb.set_trace()
        print(str(i) + "th item" + " prediction: " + str(round(predict_val, 5))+ " real ctr : " + str(round(real_ctr, 5)))


    print('Inference done')
    return full_pred

def load_model(model, filename):
    state = torch.load(filename)
    model.load_state_dict(state['model'])
    print("model is loaded!")

# Save model into the file
def save_model(model, filename):
    state = { 'model': model.state_dict() }
    torch.save(state, filename)
    print("model is saved!")


def save_csv(result_fname, data_csv_fname, test_loader, results):
    assert(test_loader.batch_size == 1)
    data_csv = pd.read_csv(data_csv_fname)
    with open(result_fname, 'w') as f:
        f.write('file_name,bad,good,ctr\n')
        for i, data in enumerate(test_loader):
            if i == 0: continue # CSV header absence
            _, _, ctrs, _, _, _ = data
            prob = results[i]
            name = data_csv.iloc[i-1, 0]
            assert abs(ctrs.item() / 100 - data_csv.iloc[i-1, 3]) < 1e-2, 'Ctr is different: %f, %f' % (ctrs.item()/100, data_csv.iloc[i-1, 3])
            f.write('%s,%f,%f,%f\n'%(name, prob[0].item(), prob[1].item(), ctrs.item() / 100))

def preprocess(output_folder, args):
    import os
    output_folder = output_folder[0]
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    transforms = get_transforms(args)['train']
    train_dataset = CreativeDataset(os.path.join(DATASET_PATH, 'train', 'train') + '/train_data',
                                     os.path.join(DATASET_PATH, 'train'), args, transforms)
    loader = torch.utils.data.DataLoader(train_dataset,
                                            shuffle=False,
                                            batch_size=1,
                                            num_workers=2)
    for i, data in enumerate(loader):
        image, labels, ctrs, sexs, age_areas, adids = data
        adname = train_dataset.item.iloc[i, 0].split('/')[-1]
        torch.save(image.squeeze(), output_folder+adname+'.pt')    

def get_forward_result(model, loader, args):
    result, correct, idx = [], 0.0, 0
    with torch.no_grad():
        model.train(False)  # Set model to evaluate mode
        for i, data in enumerate(loader):

            inputs, labels, ctrs, sexs, age_areas, adids = data
            if args.use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
                ctrs = Variable(ctrs.float().cuda())
                sexs = Variable(sexs.float().cuda())
                age_areas = Variable(age_areas.float().cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
                ctrs, sexs, age_areas = Variable(ctrs.float()), Variable(sexs.float()), Variable(age_areas.float())
            
            result.append(model(inputs, sexs, age_areas))
            correct += torch.sum(labels == torch.max(result[-1], 1)[1])
            idx += labels.size()[0]
    return result, (correct.float() / len(loader)).item()