import glob
from .ImageListDataset import ImageListDataset
from torchvision import transforms
from torch.utils.data import DataLoader


def setup_dataloader(name, h=128, w=128, batch_size=4, num_workers=4, data_size=30):
    '''
    instead of setting up dataloader that read raw image from file, 
    let's use store all images on cpu memmory
    because this is for small dataset
    '''

    img_path_dict = {}
    if name == "animal":
        labels_dict = {'cat': 0, 'dog': 1, 'wild': 2}
        for label in labels_dict.keys():
            img_path_dict[label] = glob.glob(f"data/afhq/train/{label}/*.jpg")
            img_path_dict[label] = img_path_dict[label][:data_size]
            print(img_path_dict[label][0])
    else:
        raise NotImplementedError("Unknown dataset %s" % name)

    transform = transforms.Compose([
        transforms.Resize(min(h, w)),
        transforms.CenterCrop((h, w)),
        transforms.ToTensor(),
    ])

    img_path_list = []

    for label in labels_dict.keys():
        img_path_list.extend([(path, labels_dict[label])
                             for path in img_path_dict[label]])

    img_path_list = [(data[0], (i, data[1]))
                     for i, data in enumerate(sorted(img_path_list))]
    dataset = ImageListDataset(img_path_list, transform=transform)

    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=True, num_workers=num_workers, persistent_workers=True, pin_memory=True)

    #[data for data in dataset]


if __name__ == '__main__':
    setup_dataloader('animal', h=128, w=128, batch_size=4,
                     num_workers=4, data_size=30)
