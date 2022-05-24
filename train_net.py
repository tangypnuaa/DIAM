import argparse
import numpy as np
import os
import random
from tqdm import tqdm
import horovod.torch as hvd
import torch
import time
from ofa.imagenet_classification.elastic_nn.modules.dynamic_op import (
    DynamicSeparableConv2d,
)
from ofa.imagenet_classification.run_manager import DistributedImageNetALRunConfig
from ofa.imagenet_classification.run_manager.distributed_run_manager import (
    DistributedRunManager,
)
from ofa.utils import MyRandomResizedCrop
from load_test_models import load_test_models


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
    ],
)
parser.add_argument(
    "--dataset",
    type=str,
    default="mnist",
    choices=[
        "mnist",
        "kmnist",
    ],
)
parser.add_argument("--al_iter", type=int, default=0)
parser.add_argument("--net_id", type=int, default=0)

parser.add_argument("--mDIS", action="store_true")
parser.add_argument("--DIS_frac", type=float, default=0.5, help="Use the last {DIS_frac * total_epoch} "
                                                                "epochs to evaluate the unlabeled data")

args = parser.parse_args()

if args.al_iter == 0:
    SAVING_ROOT = f"exp/{args.al_iter}/{args.dataset}"
else:
    SAVING_ROOT = f"exp/{args.al_iter}/{args.dataset}/{args.method}"

args.path = os.path.join(SAVING_ROOT, "%d" % args.net_id)

args.n_epochs = 20
args.base_lr = 7.5e-3
args.warmup_epochs = 0
args.warmup_lr = -1

args.manual_seed = 0
args.lr_schedule_type = "cosine"
args.base_batch_size = 120
# serve as ini. lab. set size in AL setting. 5% of training data
args.valid_size = 3000

args.opt_type = "sgd"
args.momentum = 0.9
args.no_nesterov = False
args.weight_decay = 3e-5
args.label_smoothing = 0.1
args.no_decay_keys = "bn#bias"
args.fp16_allreduce = False

args.model_init = "he_fout"
args.validation_frequency = 5
if args.mDIS:
    args.validation_frequency = 30
args.print_frequency = 10

args.n_worker = 16
args.resize_scale = 0.08
args.distort_color = "tf"
args.continuous_size = True
args.not_sync_distributed_image_size = False

args.bn_momentum = 0.1
args.bn_eps = 1e-5
args.dropout = 0.1
args.base_stage_width = "proxyless"

args.width_mult_list = "1.0"
args.dy_conv_scaling_mode = 1
args.independent_distributed_sampling = False

args.kd_ratio = 0   # disable the kd during model training
args.kd_type = "ce"


def train_one_epoch(run_manager, args, epoch, warmup_epochs=0, warmup_lr=0):
    dynamic_net = run_manager.network
    distributed = isinstance(run_manager, DistributedRunManager)

    # switch to train mode
    dynamic_net.train()
    # if distributed:
    #     run_manager.run_config.train_loader.sampler.set_epoch(epoch)
    MyRandomResizedCrop.EPOCH = epoch
    nBatch = len(run_manager.run_config.train_loader)

    metric_dict = run_manager.get_metric_dict()

    with tqdm(
        total=nBatch,
        desc="Train Epoch #{}".format(epoch + 1),
        disable=distributed and not run_manager.is_root,
    ) as t:
        end = time.time()
        for i, (images, labels) in enumerate(run_manager.run_config.train_loader):
            MyRandomResizedCrop.BATCH = i
            if epoch < warmup_epochs:
                new_lr = run_manager.run_config.warmup_adjust_learning_rate(
                    run_manager.optimizer,
                    warmup_epochs * nBatch,
                    nBatch,
                    epoch,
                    i,
                    warmup_lr,
                )
            else:
                new_lr = run_manager.run_config.adjust_learning_rate(
                    run_manager.optimizer, epoch - warmup_epochs, i, nBatch
                )

            images, labels = images.cuda(), labels.cuda()
            target = labels

            # clean gradients
            dynamic_net.zero_grad()

            # compute output
            output = run_manager.net(images)
            tr_loss = run_manager.train_criterion(output, labels)
            loss_type = "ce"
            # measure accuracy and record loss
            run_manager.update_metric(metric_dict, output, target)

            tr_loss.backward()
            run_manager.optimizer.step()

            t.set_postfix(
                {
                    **run_manager.get_metric_vals(metric_dict, return_dict=True),
                    "R": images.size(2),
                    "lr": new_lr,
                    "loss_type": loss_type
                }
            )
            t.update(1)
            end = time.time()

    ############# evaluate unlabeled data ########################################################
    # process the params of mDIV method
    if run_manager.run_config.mDIS and epoch >= run_manager.run_config.mDIS_start_epoch:
        unlab_dataloader = run_manager.run_config.data_provider.unlab
        dynamic_net.zero_grad()
        dynamic_net.eval()
        prediction_record = torch.zeros(len(run_manager.run_config.data_provider.unlab_indexes), dtype=torch.int)
        with torch.no_grad():
            for ii, (images, labels) in enumerate(unlab_dataloader):
                images, labels = images.to("cuda"), labels.to("cuda")
                output = dynamic_net(images)
                _, pred_lab = torch.max(output.data, 1)
                prediction_record[ii*unlab_dataloader.batch_size:ii*unlab_dataloader.batch_size+len(pred_lab)] = pred_lab
        torch.save(prediction_record, os.path.join(run_manager.save_path, f"{epoch}.pt"))
    ################################################################################################
    return run_manager.get_metric_vals(metric_dict)


if __name__ == "__main__":
    os.makedirs(args.path, exist_ok=True)
    NCLASSES = 10

    hvd.init()

    net, image_size = load_test_models(net_id=args.net_id, n_classes=NCLASSES, trained_weights=None)
    args.image_size = image_size
    MyRandomResizedCrop.CONTINUOUS = args.continuous_size
    MyRandomResizedCrop.SYNC_DISTRIBUTED = not args.not_sync_distributed_image_size

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    # build run config from args
    args.lr_schedule_param = None
    args.opt_param = {
        "momentum": args.momentum,
        "nesterov": not args.no_nesterov,
    }
    args.init_lr = args.base_lr * 1  # linearly rescale the learning rate
    if args.warmup_lr < 0:
        args.warmup_lr = args.base_lr
    args.train_batch_size = args.base_batch_size
    args.test_batch_size = args.base_batch_size
    run_config = DistributedImageNetALRunConfig(
        **args.__dict__, num_replicas=1, rank=hvd.rank()
    )
    if args.al_iter > 0:
        # load updated labeled data indexes
        al_save_root = f"al_results/{args.al_iter}/{args.dataset}/{args.method}/"
        lab_idx = np.loadtxt(os.path.join(al_save_root, "lab_idx.txt"), dtype=int)
        unlab_idx = np.loadtxt(os.path.join(al_save_root, "unlab_idx.txt"), dtype=int)
        # attach to run_config
        run_config.lab_idx = lab_idx
        run_config.unlab_idx = unlab_idx
    else:
        # save idx to file
        np.savetxt(os.path.join(SAVING_ROOT, "unlab_idx.txt"), run_config.data_provider.unlab_indexes, fmt="%d")
        np.savetxt(os.path.join(SAVING_ROOT, "lab_idx.txt"), run_config.data_provider.lab_indexes, fmt="%d")

    # attach the params of multi_DIS method to run_config
    run_config.mDIS = args.mDIS
    run_config.DIS_frac = args.DIS_frac
    run_config.mDIS_start_epoch = args.n_epochs - round(args.n_epochs * run_config.DIS_frac)

    if args.dy_conv_scaling_mode == -1:
        args.dy_conv_scaling_mode = None
    DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = args.dy_conv_scaling_mode

    # """ Distributed RunManager """
    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    distributed_run_manager = DistributedRunManager(
        args.path,
        net,
        run_config,
        compression,
        backward_steps=1,
        is_root=(hvd.rank() == 0),
    )
    distributed_run_manager.save_config()
    # training
    for epoch in range(
            distributed_run_manager.start_epoch, distributed_run_manager.run_config.n_epochs + args.warmup_epochs
    ):
        train_top1, train_top5 = train_one_epoch(
            distributed_run_manager, args, epoch, args.warmup_epochs, args.warmup_lr
        )

        if (epoch + 1) % args.validation_frequency == 0:
            loss, (top1, top5) = distributed_run_manager.validate(
                epoch=epoch, is_test=True, run_str="", net=None
            )
            # best_acc
            is_best = top1 > distributed_run_manager.best_acc
            distributed_run_manager.best_acc = max(distributed_run_manager.best_acc, top1)
            if not distributed_run_manager or distributed_run_manager.is_root:
                val_log = (
                    "Valid [{0}/{1}] loss={2:.3f}, top-1={3:.3f} ({4:.3f})".format(
                        epoch + 1 - args.warmup_epochs,
                        distributed_run_manager.run_config.n_epochs,
                        loss,
                        top1,
                        distributed_run_manager.best_acc,
                    )
                )
                val_log += ", Train top-1 {top1:.3f}\t".format(
                    top1=train_top1
                )
                distributed_run_manager.write_log(val_log, "valid", should_print=False)

                if distributed_run_manager.run_config.mDIS:
                    print("mDIS for AL, skip saving models")
                else:
                    print(f"save model to {distributed_run_manager.save_path}")
                    distributed_run_manager.save_model(
                        {
                            "epoch": epoch,
                            "best_acc": distributed_run_manager.best_acc,
                            "optimizer": distributed_run_manager.optimizer.state_dict(),
                            "state_dict": distributed_run_manager.network.state_dict(),
                        },
                        is_best=is_best,
                    )
