from TorchDB_mod import DBreader_frame_interpolation

from torch.utils.data import DataLoader

data_root = "F:/Pycharm Projects/pytorch-sepconv/db"
resize = (128, 128)
batch_size = 1

dataset = DBreader_frame_interpolation(data_root, resize=resize, is_training=True)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

for batch_idx, batch in enumerate(train_loader):
    print("Batch", batch_idx)
    print("Batch size:", len(batch))
    #1 sample chua 3 hinh -> 1 indx trong batch co 3 sample
    for sample_idx, sample in enumerate(batch):
        print("Sample", sample_idx)
        print("Sample shape:", sample.shape)