import argparse
import os

import alipy
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import preprocessing
from tqdm import tqdm

from load_test_models import load_test_models
from ofa.imagenet_classification.networks import MobileNetV3Large
from ofa.imagenet_classification.run_manager.run_config import get_data_provider_by_name

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
parser.add_argument("--batch_size", type=int, default=1500)
parser.add_argument("--save_root", type=str, default="./al_results")
parser.add_argument("--al_iter", type=int, default=0)
parser.add_argument("--model_num", type=int, default=12)

args = parser.parse_args()
if args.al_iter == 0:
    ofa_checkpoint_root = f"exp/{args.al_iter}/{args.dataset}/"
else:
    ofa_checkpoint_root = f"exp/{args.al_iter}/{args.dataset}/{args.method}/"
al_idx_save_root = os.path.join(args.save_root, str(args.al_iter), args.dataset)
al_save_root = os.path.join(al_idx_save_root, args.method)
os.makedirs(al_save_root, exist_ok=True)
NCLASSES = 10


class ALScoringFunctions:
    def __init__(self, batch_size, n_classes, unlab_idx, net, dataloader):
        self.net = net
        self.dataloader = dataloader
        self.scores = []
        self.unlab_idx = unlab_idx
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.probs = None
        self.embeddings = None

    def _get_proba_pred(self):
        self.net.eval()
        with torch.no_grad():
            probs = torch.zeros([len(unlab_idx), self.n_classes])
            embeddings = None
            with tqdm(total=len(self.dataloader), desc="Extracting unlabeled data embedding/proba prediction...") as t:
                for i, (images, labels) in enumerate(self.dataloader):
                    images, labels = images.to("cuda"), labels.to("cuda")
                    logits, output = self.net.get_logits_and_pred(images)
                    prob = F.softmax(output, dim=1)
                    probs[i * self.dataloader.batch_size:i * self.dataloader.batch_size + len(labels),
                    :] = prob.detach()
                    if embeddings is None:
                        embeddings = logits.detach()
                    else:
                        embeddings = torch.cat([embeddings, logits.detach()], dim=0)
                    t.update(1)
        self.probs = probs
        self.embeddings = embeddings

    def margin(self, return_scores=False):
        if self.probs is None:
            self._get_proba_pred()
        probs_sorted, idxs = self.probs.sort(descending=True)
        uncertainties = probs_sorted[:, 0] - probs_sorted[:, 1]
        if return_scores:
            return uncertainties  # smaller is better
        else:
            return self.unlab_idx[uncertainties.sort()[1][:self.batch_size]]  # sort: small -> large

    def entropy(self, return_scores=False):
        if self.probs is None:
            self._get_proba_pred()
        log_probs = torch.log(self.probs)
        uncertainties = (self.probs * log_probs).sum(1)
        if return_scores:
            return uncertainties  # smaller is better
        else:
            return self.unlab_idx[uncertainties.sort()[1][:self.batch_size]]  # double negative, take the smallest ones

    def least_conf(self, return_scores=False):
        if self.probs is None:
            self._get_proba_pred()
        uncertainties = self.probs.max(1)[0]
        if return_scores:
            return uncertainties  # smaller is better
        else:
            return self.unlab_idx[uncertainties.sort()[1][:self.batch_size]]

    def coreset(self, lab_dataloader, return_scores=False):
        if self.embeddings is None:
            self._get_proba_pred()
        # get labeled data embeddings
        self.net.eval()
        with torch.no_grad():
            lab_embeddings = None
            with tqdm(total=len(lab_dataloader), desc="Extracting labeled data embedding...") as t:
                for i, (images, labels) in enumerate(lab_dataloader):
                    images, labels = images.to("cuda"), labels.to("cuda")
                    logits = self.net.get_logits(images).detach()
                    if lab_embeddings is None:
                        lab_embeddings = logits.detach()
                    else:
                        lab_embeddings = torch.cat([lab_embeddings, logits.detach()], dim=0)
                    t.update(1)

        lab_num = lab_embeddings.shape[0]
        unlab_num = self.embeddings.shape[0]
        coreset_qs = alipy.query_strategy.QueryInstanceCoresetGreedy(
            X=torch.cat([self.embeddings.cpu(), lab_embeddings.cpu()], dim=0), y=np.zeros(lab_num + unlab_num),
            train_idx=list(range(lab_num + unlab_num)))
        if return_scores:
            selected = coreset_qs.select(label_index=np.arange(lab_num) + unlab_num,
                                         unlabel_index=np.arange(unlab_num), batch_size=len(self.unlab_idx))
            scores = np.zeros(len(self.unlab_idx))
            for iidx, idx in enumerate(selected):
                scores[idx] = iidx  # smaller is better
            return scores
        else:
            selected = coreset_qs.select(label_index=np.arange(lab_num) + unlab_num,
                                         unlabel_index=np.arange(unlab_num), batch_size=self.batch_size)
            return [self.unlab_idx[i] for i in selected]

    def CAL(self, return_scores=False):
        global ofa_checkpoint_root
        flist = os.listdir(os.path.join(ofa_checkpoint_root, '0', "checkpoint"))
        pt_list = [fi for fi in flist if fi.endswith('.pt')]
        pt_list = sorted(pt_list)
        all_votes = torch.zeros([len(pt_list), args.model_num, len(self.unlab_idx)])
        for ipt, pt in enumerate(pt_list):
            for i in range(args.model_num):
                ofa_unlab_pred_root = os.path.join(ofa_checkpoint_root, str(i), "checkpoint")
                pre_mat = torch.load(os.path.join(ofa_unlab_pred_root, pt))
                all_votes[ipt, i] = pre_mat
        first_order = []
        for voter in range(args.model_num):
            votes = all_votes[:, voter, :]
            is_dis = (torch.amax(votes, dim=0) != torch.amin(votes, dim=0))
            first_order.append(is_dis.numpy())
        cand_args = np.where(np.sum(first_order, axis=0) != 0)[0]
        np.random.shuffle(cand_args)
        selected = cand_args[:self.batch_size]
        return [self.unlab_idx[i] for i in selected]

    def DIAM(self, return_scores=False):
        global ofa_checkpoint_root
        # ofa_unlab_pred_root = os.path.join(ofa_checkpoint_root, '0', "checkpoint")
        flist = os.listdir(os.path.join(ofa_checkpoint_root, '0', "checkpoint"))
        pt_list = [fi for fi in flist if fi.endswith('.pt')]
        pt_list = sorted(pt_list)
        all_votes = torch.zeros([len(pt_list), args.model_num, len(self.unlab_idx)])
        for ipt, pt in enumerate(pt_list):
            for i in range(args.model_num):
                ofa_unlab_pred_root = os.path.join(ofa_checkpoint_root, str(i), "checkpoint")
                pre_mat = torch.load(os.path.join(ofa_unlab_pred_root, pt))
                all_votes[ipt, i] = pre_mat
        first_order = []
        second_order = []
        for voter in range(args.model_num):
            votes = all_votes[:, voter, :]
            is_dis = (torch.amax(votes, dim=0) != torch.amin(votes, dim=0))
            first_order.append(is_dis.numpy())
            unc = alipy.query_strategy.QueryInstanceQBC.calc_vote_entropy(votes.numpy())
            second_order.append(unc)
        sorted_args = np.lexsort((np.sum(second_order, axis=0), np.sum(first_order, axis=0)))[::-1]
        selected = sorted_args[:self.batch_size * 5]
        np.random.shuffle(selected)
        selected = selected[:self.batch_size]
        return [self.unlab_idx[i] for i in selected]


if __name__ == "__main__":
    if args.al_iter == 0:
        lab_idx = np.loadtxt(os.path.join(ofa_checkpoint_root, "lab_idx.txt"), dtype=int)
        unlab_idx = np.loadtxt(os.path.join(ofa_checkpoint_root, "unlab_idx.txt"), dtype=int)
    else:
        al_idx_save_root = os.path.join(args.save_root, str(args.al_iter), args.dataset)
        al_save_root = os.path.join(al_idx_save_root, args.method)
        lab_idx = np.loadtxt(os.path.join(al_save_root, "lab_idx.txt"), dtype=int)
        unlab_idx = np.loadtxt(os.path.join(al_save_root, "unlab_idx.txt"), dtype=int)
    if args.method == "random":
        np.random.shuffle(unlab_idx)
        selected_idx = unlab_idx[:args.batch_size]
    else:
        # load model
        net, image_size = load_test_models(net_id=0, trained_weights=None)
        del net
        ########################### construct sequential unlab load dataloader ########################################
        DataProvider = get_data_provider_by_name(args.dataset)
        dpv = DataProvider(
            train_batch_size=256,
            test_batch_size=256,
            valid_size=3000,
            n_worker=16,
            resize_scale=0.08,
            distort_color="tf",
            image_size=image_size,
            num_replicas=None,
            lab_idx=lab_idx,
            unlab_idx=unlab_idx,
        )
        unlab_dataloader = dpv.unlab
        ######################### query ##########################################
        if args.method == "coreset":
            lab_dataloader = dpv.train
            teacher_model = MobileNetV3Large(
                n_classes=NCLASSES,
                bn_param=(0.1, 1e-5),
                dropout_rate=0,
                width_mult=1.0,
                ks=7,
                expand_ratio=6,
                depth_param=4,
            )
            teacher_model.load_state_dict(
                torch.load(".torch/ofa_checkpoints/0/ofa_D4_E6_K7", map_location="cpu")["state_dict"])
            teacher_model.cuda()
            qs = ALScoringFunctions(batch_size=args.batch_size,
                                    n_classes=NCLASSES,
                                    unlab_idx=unlab_idx,
                                    net=teacher_model,
                                    dataloader=unlab_dataloader)
            selected_idx = qs.coreset(lab_dataloader=lab_dataloader)
        elif args.method != "DIAM" and args.method != "CAL":
            minmax = preprocessing.MinMaxScaler()
            all_scores = np.zeros([args.model_num, len(unlab_idx)])
            for rs in tqdm(range(args.model_num), desc="test models"):
                ofa_net_weight = os.path.join(ofa_checkpoint_root, str(rs), "checkpoint/checkpoint.pth.tar")
                net, image_size = load_test_models(net_id=rs, n_classes=NCLASSES, trained_weights=ofa_net_weight)
                net.cuda()
                qs = ALScoringFunctions(batch_size=args.batch_size,
                                        n_classes=NCLASSES,
                                        unlab_idx=unlab_idx,
                                        net=net,
                                        dataloader=unlab_dataloader)
                kwargs = dict()
                kwargs['return_scores'] = True
                data_socres = eval(f"qs.{args.method}(**kwargs)")
                all_scores[rs, :] = minmax.fit_transform(
                    data_socres.numpy().reshape([1, -1]))  # scale to 0-1, avoid dominating by a certain qs

            # mean and sort the scores
            mean_all_scores = np.mean(all_scores, axis=0)
            selected_idx = np.argsort(mean_all_scores)[:args.batch_size]
            selected_idx = [unlab_idx[i] for i in selected_idx]
        else:
            qs = ALScoringFunctions(batch_size=args.batch_size,
                                    n_classes=NCLASSES,
                                    unlab_idx=unlab_idx,
                                    net=None,
                                    dataloader=unlab_dataloader)
            selected_idx = eval(f"qs.{args.method}()")

    # update index
    # test validate
    assert set(selected_idx).issubset(set(unlab_idx))
    lab_idx = np.hstack((lab_idx, selected_idx))
    unlab_idx = np.setdiff1d(unlab_idx, lab_idx)
    np.random.shuffle(unlab_idx)
    # save
    al_idx_save_root = os.path.join(args.save_root, str(args.al_iter + 1), args.dataset)
    al_save_root = os.path.join(al_idx_save_root, args.method)
    os.makedirs(al_save_root, exist_ok=True)
    print(f"save to {al_save_root}...")
    np.savetxt(os.path.join(al_save_root, "lab_idx.txt"), lab_idx, fmt="%d")
    np.savetxt(os.path.join(al_save_root, "unlab_idx.txt"), unlab_idx, fmt="%d")
