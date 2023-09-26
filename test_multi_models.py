import os
import numpy as np
import random
import torch
import argparse
from tqdm import tqdm
import torch.nn.functional as F

from load_test_models import load_test_models
from ofa.utils import cross_entropy_loss_with_soft_target
from ofa.imagenet_classification.run_manager.run_config import get_data_provider_by_name
from ofa.utils import accuracy


parser = argparse.ArgumentParser()
parser.add_argument(
    "--method",
    type=str,
    default="DIAM",
    choices=[
        "margin",
        "entropy",
        "least_conf",
        "coreset",
        "random",
        "CAL",
        "DIAM",
        "DIAM_qp",
    ],
)
parser.add_argument(
    "--dataset",
    type=str,
    default="mnist",
    choices=[
        "mnist",
        "kmnist",
        "fmnist",
        "emnistlet",
        "emnistdig",
    ],
)
parser.add_argument("--al_iter", type=int, default=0)
parser.add_argument("--model_num", type=int, default=12)
args = parser.parse_args()


if __name__ == "__main__":
    results_saving_dir = f"./extracted_results/{args.model_num}/{args.dataset}/{args.method}/{args.al_iter}"
    print("testing ", results_saving_dir)
    if os.path.exists(os.path.join(results_saving_dir, "performances.txt")):
        exit(0)
    os.makedirs(results_saving_dir, exist_ok=True)
    NCLASSES = 10

    if args.al_iter == 0:
        ofa_checkpoint_root = f"./exp/{args.al_iter}/{args.dataset}/"
        lab_idx = np.loadtxt(os.path.join(ofa_checkpoint_root, "lab_idx.txt"), dtype=int)
        unlab_idx = np.loadtxt(os.path.join(ofa_checkpoint_root, "unlab_idx.txt"), dtype=int)
    else:
        ofa_checkpoint_root = f"./exp/{args.al_iter}/{args.dataset}/{args.method}/"
        al_idx_save_root = os.path.join(f"./al_results", str(args.al_iter), args.dataset)
        al_save_root = os.path.join(al_idx_save_root, args.method)
        lab_idx = np.loadtxt(os.path.join(al_save_root, "lab_idx.txt"), dtype=int)
        unlab_idx = np.loadtxt(os.path.join(al_save_root, "unlab_idx.txt"), dtype=int)
    # load model
    net, image_size = load_test_models(net_id=0, trained_weights=None)
    del net
    ########################### construct sequential unlab load dataloader ########################################
    DataProvider = get_data_provider_by_name(args.dataset)
    dpv = DataProvider(
        train_batch_size=256,
        test_batch_size=128,
        valid_size=3000,
        n_worker=16,
        resize_scale=0.08,
        distort_color="tf",
        image_size=image_size,
        num_replicas=None,
        lab_idx=lab_idx,
        unlab_idx=unlab_idx,
    )
    unlab_dl = dpv.unlab
    test_dl = dpv.test
    train_dl = dpv.train

    """ Test sampled subnet 
    """
    performances = np.zeros([args.model_num, 2])
    for imodel in range(args.model_num):
        ofa_ckpt_path = os.path.join(ofa_checkpoint_root, str(imodel), "checkpoint/checkpoint.pth.tar")
        net, image_size = load_test_models(net_id=imodel, n_classes=NCLASSES, trained_weights=ofa_ckpt_path)
        net.to("cuda")
        net.eval()

        top1 = 0
        top5 = 0
        all_num = 0
        with torch.no_grad():
            with tqdm(
                    total=len(test_dl),
                    desc="Validate model #{} ".format(imodel)
            ) as t:
                for i, (images, labels) in enumerate(test_dl):
                    images, labels = images.cuda(), labels.cuda()
                    # compute output
                    output = net(images)
                    acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                    all_num += output.size(0)
                    top1 += acc1 * output.size(0)
                    top5 += acc5 * output.size(0)
                    t.update(1)

        top1 /= all_num
        top5 /= all_num
        print(top1, top5)
        performances[imodel, :] = [float(top1), float(top5)]
    np.savetxt(os.path.join(results_saving_dir, "performances.txt"), performances, fmt="%f")
