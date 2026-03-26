from torch.utils.data import DataLoader
from network.losses import *
from network.datasets.loveda_dataset import *
from network.models.S2DCB_R18 import S2DCB_R18
from catalyst.contrib.nn import Lookahead
from catalyst import utils
from network.models.unet_seg.unet_segdiff_mf import UNetModel
# training hparam
max_epoch = 30
ignore_index = len(CLASSES)
train_batch_size = 16
val_batch_size = 16
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "s2dcb_r18"
weights_path = "model_weights/loveda/{}".format(weights_name)
test_weights_name = "s2dcb_r18"
log_name = 'loveda/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None # the path for the pretrained model weight
gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None #whether continue training with the checkpoint, default None
# strategy = None

#  define the network
net = S2DCB_R18(num_classes=num_classes)

s1_net = UNetModel(
        in_channels=7,
        model_channels=64,
        out_channels=7,
        num_res_blocks=2, #2
        attention_resolutions=[2], #[2]
        dropout=0.1,
        channel_mult=[1, 2, 4, 4],
        conv_resample=False,
        dims=2,
        num_classes=None,
        d_cond_image=7,
        use_checkpoint=False,
        num_heads=2,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_new_attention_order=True,
        with_fourier_features=False,
        ignore_time=False,
        input_projection=True
    )
s1_net_path = 'model_weights/loveda/s1_net/mf.ckpt'

# define the loss
loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = True

# define the dataloader

def get_training_transform():
    train_transform = [
        albu.RandomRotate90(p=0.5),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=ignore_index, nopad=False)])
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


train_dataset = LoveDATrainDataset(transform=train_aug, data_root='data/LoveDA/train_val')

val_dataset = loveda_val_dataset

test_dataset = LoveDATestDataset()

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)

