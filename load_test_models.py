import torch.utils.data

from ofa.utils import download_url
from ofa.model_zoo import ofa_specialized

test_models = [
    ################# Samsung note10 #################
    "note10_lat@22ms_top1@76.6_finetune@25",
    "note10_lat@16ms_top1@75.5_finetune@25",
    "note10_lat@11ms_top1@73.6_finetune@25",
    "note10_lat@8ms_top1@71.4_finetune@25",
    ################# Samsung note8 #################
    "note8_lat@65ms_top1@76.1_finetune@25",
    "note8_lat@49ms_top1@74.9_finetune@25",
    "note8_lat@31ms_top1@72.8_finetune@25",
    "note8_lat@22ms_top1@70.4_finetune@25",
    ################# Samsung S7 Edge #################
    "s7edge_lat@88ms_top1@76.3_finetune@25",
    "s7edge_lat@58ms_top1@74.7_finetune@25",
    "s7edge_lat@41ms_top1@73.1_finetune@25",
    "s7edge_lat@29ms_top1@70.5_finetune@25",
]


def load_test_models(net_id: int = 0, n_classes=10, trained_weights=None):
    net, image_size = ofa_specialized(net_id=test_models[net_id], n_classes=n_classes, pretrained=False)
    if trained_weights:
        init = torch.load(
            trained_weights,
            map_location="cpu",
        )["state_dict"]
        net.load_state_dict(init)
    else:
        url_base = "https://hanlab.mit.edu/files/OnceForAll/ofa_specialized/"
        init = torch.load(
            download_url(
                url_base + test_models[net_id] + "/init",
                model_dir=".torch/ofa_specialized/%s/" % test_models[net_id],
            ),
            map_location="cpu",
        )["state_dict"]
        net.load_state_dict(init)
    return net, image_size
