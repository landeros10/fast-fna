'''
Created May 11, 2020
author: landeros10

Center for Systems Biology
Massachusetts General Hospital
'''
from __future__ import division, print_function
from unet_inj.unet import Unet_Inj
from unet_inj.util import get_num_params


NET_KWARGS = dict(input_s=300,
                  rate=0.00,
                  layers=3,
                  feat_factor=3,
                  features_root=20,
                  filter_size=3,
                  dapiFeat=20, inj="second", joinType="concat")


GLOBAL_BATCH_SIZE = 20

EPOCHS = 500
BESTACC = 98
RESTORE = False
LR = 1e-2

TRAIN_KWARGS = {"epochs": EPOCHS,
                "display_step": 10,
                "store_every": 25,
                "store_size": 20,
                "store_resize_f": 0.5,
                "bestAcc": BESTACC,
                "bs": GLOBAL_BATCH_SIZE,
                "learning_rate": LR,
                "resize_lims": [0.75, 1.0],
                "pred_path": "",
                "out_path": "",
                "log_path": ""}

LOSS_KWARGS = {"cost_type": "combined",
               "dice_coeff": 900,
               "class_weights": [1, 1, 1]}
N_CLASS = 3

if __name__ == "__main__":
    unet_inj_model = Unet_Inj(n_class=N_CLASS,
                              net_kwargs=NET_KWARGS)

    num_params = get_num_params(unet_inj_model)
    print("\n\nTrainable parameters: ", num_params)

    unet_inj_model.train("", "",
                         restore=RESTORE,
                         train_kwargs=TRAIN_KWARGS,
                         loss_kwargs=LOSS_KWARGS)
