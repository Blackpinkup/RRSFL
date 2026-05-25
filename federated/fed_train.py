"""
RRSFL: Redundancy-Restricted Shared Feature Learning.
"""
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import argparse
import copy
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms

from utils.dataset import Camelyon17, DGDR, Prostate
from utils.loss import DiceLoss, JointLoss
from nets.models import DenseNet, UNet


def unpack_output(model_output):
    if isinstance(model_output, tuple):
        return model_output[0], model_output[1], model_output[2]
    return model_output, None, None


def train_one_client(args, model, data_loader, optimizer, loss_fun, device):
    model.to(device)
    model.train()

    loss_all = 0.0
    task_loss_all = 0.0
    icw_loss_all = 0.0
    total = 0
    correct = 0
    train_acc = 0.0
    segmentation = model.__class__.__name__ == "UNet"
    log_interval = max(1, math.ceil(len(data_loader) * 0.2))

    for step, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()

        data = data.to(device)
        target = target.to(device)
        output, _, icw_loss = unpack_output(
            model(data, return_embeddings=False, return_icw=not args.disable_icw)
        )
        task_loss = loss_fun(output, target)
        if icw_loss is None:
            icw_loss = output.new_zeros(())
        loss = task_loss + args.lambda_icw * icw_loss

        loss.backward()
        optimizer.step()

        loss_all += loss.item()
        task_loss_all += task_loss.item()
        icw_loss_all += float(icw_loss.detach().item())

        if segmentation:
            train_acc += DiceLoss().dice_coef(output.detach(), target).item()
        else:
            total += target.size(0)
            pred = output.detach().max(1)[1]
            batch_correct = pred.eq(target.view(-1)).sum().item()
            correct += batch_correct
            if step % log_interval == 0:
                print(
                    " [Step-{}|{}]| Train Loss: {:.4f} | Train Acc: {:.4f}".format(
                        step, len(data_loader), loss.item(), batch_correct / target.size(0)
                    ),
                    end="\r",
                )

    loss = loss_all / max(1, len(data_loader))
    task_loss = task_loss_all / max(1, len(data_loader))
    icw_loss = icw_loss_all / max(1, len(data_loader))
    acc = train_acc / max(1, len(data_loader)) if segmentation else correct / max(1, total)

    model.to("cpu")
    return loss, task_loss, icw_loss, acc


def test(args, model, data_loader, loss_fun, device):
    model.to(device)
    model.eval()
    loss_all = 0.0
    total = 0
    correct = 0
    test_acc = 0.0
    segmentation = model.__class__.__name__ == "UNet"
    log_interval = max(1, math.ceil(len(data_loader) * 0.2))

    with torch.no_grad():
        for step, (data, target) in enumerate(data_loader):
            data = data.to(device)
            target = target.to(device)
            output, _, _ = unpack_output(model(data, return_embeddings=False, return_icw=False))
            loss = loss_fun(output, target)
            loss_all += loss.item()

            if segmentation:
                test_acc += DiceLoss().dice_coef(output, target).item()
            else:
                total += target.size(0)
                pred = output.max(1)[1]
                batch_correct = pred.eq(target.view(-1)).sum().item()
                correct += batch_correct
                if step % log_interval == 0:
                    print(
                        " [Step-{}|{}]| Test Acc: {:.4f}".format(
                            step, len(data_loader), batch_correct / target.size(0)
                        ),
                        end="\r",
                    )

    loss = loss_all / max(1, len(data_loader))
    acc = test_acc / max(1, len(data_loader)) if segmentation else correct / max(1, total)
    model.to("cpu")
    return loss, acc


def sfe_loss_from_embeddings(client_embeddings):
    if not client_embeddings:
        raise ValueError("SFE requires at least one client embedding set.")

    stage_num = len(client_embeddings[0])
    loss = client_embeddings[0][0].new_zeros(())
    term_count = 0
    for stage_idx in range(stage_num):
        stage_embeddings = [client_embeddings[c][stage_idx] for c in range(len(client_embeddings))]
        for c_idx in range(len(stage_embeddings)):
            for c_jdx in range(c_idx + 1, len(stage_embeddings)):
                loss = loss + F.mse_loss(stage_embeddings[c_idx], stage_embeddings[c_jdx], reduction="mean")
                term_count += 1
    return loss / max(1, term_count)


def collect_client_feature_means(model, data_loader, device):
    model.to(device)
    model.eval()
    feature_sums = None
    sample_count = 0

    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            stage_vectors = model.extract_stage_vectors(data)
            batch_size = data.size(0)
            if feature_sums is None:
                feature_sums = [vec.sum(dim=0) for vec in stage_vectors]
            else:
                for idx, vec in enumerate(stage_vectors):
                    feature_sums[idx] = feature_sums[idx] + vec.sum(dim=0)
            sample_count += batch_size

    if feature_sums is None:
        raise RuntimeError("Cannot compute RRSFL client embeddings from an empty data loader.")

    means = [(feat_sum / sample_count).detach().cpu() for feat_sum in feature_sums]
    model.to("cpu")
    return means


def optimize_sfe_and_get_embeddings(args, models, train_loaders, projection_optimizers, device):
    feature_means = [
        collect_client_feature_means(model, train_loader, device)
        for model, train_loader in zip(models, train_loaders)
    ]

    sfe_loss_value = 0.0
    if not args.disable_sfe:
        for _ in range(args.sfe_steps):
            for optimizer in projection_optimizers:
                optimizer.zero_grad()

            client_embeddings = []
            for model, means in zip(models, feature_means):
                model.to(device)
                device_means = [mean.to(device) for mean in means]
                embeddings = model.project_stage_vectors([mean.unsqueeze(0) for mean in device_means])
                client_embeddings.append([emb.squeeze(0) for emb in embeddings])

            sfe_loss = sfe_loss_from_embeddings(client_embeddings)
            sfe_loss.backward()
            for optimizer in projection_optimizers:
                optimizer.step()
            sfe_loss_value = float(sfe_loss.detach().item())

    client_embeddings = []
    with torch.no_grad():
        for model, means in zip(models, feature_means):
            model.to(device)
            device_means = [mean.to(device) for mean in means]
            embeddings = model.project_stage_vectors([mean.unsqueeze(0) for mean in device_means])
            client_embeddings.append([emb.squeeze(0).detach().cpu() for emb in embeddings])
            model.to("cpu")

    return client_embeddings, sfe_loss_value


def compute_stage_weights(args, client_embeddings, device):
    client_num = len(client_embeddings)
    stage_num = len(client_embeddings[0])
    if args.disable_sfl:
        uniform = torch.full((client_num,), 1.0 / client_num, device=device)
        return [uniform.clone() for _ in range(stage_num)]

    stage_weights = []
    for stage_idx in range(stage_num):
        embeddings = torch.stack(
            [client_embeddings[c_idx][stage_idx].float().to(device).view(-1) for c_idx in range(client_num)],
            dim=0,
        )
        centroid = embeddings.mean(dim=0, keepdim=True)
        similarities = F.cosine_similarity(embeddings, centroid.expand_as(embeddings), dim=1, eps=1e-12)
        if args.nonnegative_sfl_weights:
            similarities = torch.clamp(similarities, min=0.0)
        denom = similarities.sum()
        if not torch.isfinite(denom) or torch.abs(denom).item() < 1e-12:
            weights = torch.full((client_num,), 1.0 / client_num, device=device)
        else:
            weights = similarities / denom
        stage_weights.append(weights)
    return stage_weights


def communication(args, server_model, models, client_embeddings, device):
    stage_weights = compute_stage_weights(args, client_embeddings, device)
    server_state = server_model.state_dict()
    client_states = [model.state_dict() for model in models]

    with torch.no_grad():
        for key in server_state.keys():
            stage_idx = server_model.stage_for_state_key(key)
            if stage_idx is None:
                continue

            if not torch.is_floating_point(server_state[key]):
                server_state[key].copy_(client_states[0][key])
                continue

            temp = torch.zeros_like(server_state[key], device=device)
            for client_idx, client_state in enumerate(client_states):
                weight = stage_weights[stage_idx][client_idx].to(device=device, dtype=temp.dtype)
                temp = temp + weight * client_state[key].to(device)
            server_state[key].copy_(temp.to(server_state[key].device))

        for model in models:
            model_state = model.state_dict()
            for key in server_state.keys():
                if server_model.stage_for_state_key(key) is None:
                    continue
                model_state[key].copy_(server_state[key])

    return server_model, models, [weights.detach().cpu().numpy().tolist() for weights in stage_weights]


def classification_transform(train, image_size):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def initialize(args):
    train_loaders, val_loaders, test_loaders = [], [], []
    trainsets, valsets, testsets = [], [], []
    data_name = args.data.lower()

    if data_name == "camelyon17":
        args.image_size = args.image_size or 224
        model = DenseNet(
            input_shape=[3, args.image_size, args.image_size],
            num_classes=2,
            use_amp_norm=args.use_amp_norm,
            icw_groups=args.icw_groups,
        )
        loss_fun = nn.CrossEntropyLoss()
        sites = ["1", "2", "3", "4", "5"]
        for site in sites:
            trainset_full = Camelyon17(
                site=site,
                split="train",
                transform=classification_transform(train=True, image_size=args.image_size),
            )
            testset = Camelyon17(
                site=site,
                split="test",
                transform=classification_transform(train=False, image_size=args.image_size),
            )
            val_len = math.floor(len(trainset_full) * 0.2)
            train_idx = list(range(len(trainset_full)))[:-val_len]
            val_idx = list(range(len(trainset_full)))[-val_len:]
            trainset = torch.utils.data.Subset(trainset_full, train_idx)
            valset = copy.deepcopy(torch.utils.data.Subset(trainset_full, val_idx))
            valset.dataset.transform = classification_transform(train=False, image_size=args.image_size)
            print(f"[Client {site}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}")
            trainsets.append(trainset)
            valsets.append(valset)
            testsets.append(testset)

    elif data_name == "dgdr":
        args.image_size = args.image_size or 224
        model = DenseNet(
            input_shape=[3, args.image_size, args.image_size],
            num_classes=5,
            use_amp_norm=args.use_amp_norm,
            icw_groups=args.icw_groups,
        )
        loss_fun = nn.CrossEntropyLoss()
        sites = ["APTOS", "DEEPDR", "FGADR", "IDRID", "MESSIDOR", "RLDR"]
        for site in sites:
            trainset_full = DGDR(
                site=site,
                split="train",
                transform=classification_transform(train=True, image_size=args.image_size),
            )
            try:
                valset = DGDR(
                    site=site,
                    split="val",
                    transform=classification_transform(train=False, image_size=args.image_size),
                )
                trainset = trainset_full
            except FileNotFoundError:
                val_len = math.floor(len(trainset_full) * 0.2)
                train_idx = list(range(len(trainset_full)))[:-val_len]
                val_idx = list(range(len(trainset_full)))[-val_len:]
                trainset = torch.utils.data.Subset(trainset_full, train_idx)
                valset = copy.deepcopy(torch.utils.data.Subset(trainset_full, val_idx))
                valset.dataset.transform = classification_transform(train=False, image_size=args.image_size)
            testset = DGDR(
                site=site,
                split="test",
                transform=classification_transform(train=False, image_size=args.image_size),
            )
            print(f"[Client {site}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}")
            trainsets.append(trainset)
            valsets.append(valset)
            testsets.append(testset)

    elif data_name == "prostate":
        args.image_size = args.image_size or 384
        model = UNet(
            input_shape=[3, args.image_size, args.image_size],
            init_features=args.unet_init_features,
            use_amp_norm=args.use_amp_norm,
            icw_groups=args.icw_groups,
        )
        loss_fun = JointLoss()
        sites = ["BIDMC", "HK", "I2CVB", "ISBI", "ISBI_1.5", "UCL"]
        transform = transforms.Compose([transforms.ToTensor()])

        for site in sites:
            trainset = Prostate(site=site, split="train", transform=transform)
            valset = Prostate(site=site, split="val", transform=transform)
            testset = Prostate(site=site, split="test", transform=transform)
            print(f"[Client {site}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}")
            trainsets.append(trainset)
            valsets.append(valset)
            testsets.append(testset)
    else:
        raise ValueError(f"Unsupported dataset: {args.data}")

    if args.icw_groups <= 0:
        args.icw_groups = len(trainsets)
        model.set_icw_groups(args.icw_groups)

    min_data_len = min([len(s) for s in trainsets])
    for idx in range(len(trainsets)):
        if args.imbalance:
            trainset = trainsets[idx]
        else:
            trainset = torch.utils.data.Subset(trainsets[idx], list(range(int(min_data_len))))

        valset = valsets[idx]
        testset = testsets[idx]
        train_loaders.append(
            torch.utils.data.DataLoader(
                trainset,
                batch_size=args.batch,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
            )
        )
        val_loaders.append(
            torch.utils.data.DataLoader(
                valset,
                batch_size=args.batch,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
            )
        )
        test_loaders.append(
            torch.utils.data.DataLoader(
                testset,
                batch_size=args.batch,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
            )
        )
    return model, loss_fun, sites, trainsets, testsets, train_loaders, val_loaders, test_loaders


def build_optimizers(args, models):
    task_optimizers = [
        optim.Adam(
            model.task_parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
        for model in models
    ]
    projection_optimizers = [
        optim.Adam(
            model.projection_parameters(),
            lr=args.sfe_lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
        for model in models
    ]
    task_schedulers = [
        optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.iters * args.wk_iters),
        )
        for optimizer in task_optimizers
    ]
    projection_schedulers = [
        optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.iters * args.wk_iters),
        )
        for optimizer in projection_optimizers
    ]
    return task_optimizers, projection_optimizers, task_schedulers, projection_schedulers


def save_checkpoint(
    path,
    server_model,
    models,
    task_optimizers,
    projection_optimizers,
    task_schedulers,
    projection_schedulers,
    best_epoch,
    best_acc,
    a_iter,
):
    model_dicts = {
        "server_model": server_model.state_dict(),
        "best_epoch": best_epoch,
        "best_acc": best_acc,
        "a_iter": a_iter,
    }
    for idx, model in enumerate(models):
        model_dicts[f"client_proj_{idx}"] = model.proj_heads.state_dict()
        model_dicts[f"optim_{idx}"] = task_optimizers[idx].state_dict()
        model_dicts[f"proj_optim_{idx}"] = projection_optimizers[idx].state_dict()
        model_dicts[f"sched_{idx}"] = task_schedulers[idx].state_dict()
        model_dicts[f"proj_sched_{idx}"] = projection_schedulers[idx].state_dict()
    torch.save(model_dicts, path)


def load_checkpoint(
    path,
    server_model,
    models,
    task_optimizers,
    projection_optimizers,
    task_schedulers,
    projection_schedulers,
    device,
):
    checkpoint = torch.load(path, map_location=device)
    server_model.load_state_dict(checkpoint["server_model"], strict=False)
    for client_idx, model in enumerate(models):
        model.load_state_dict(checkpoint["server_model"], strict=False)
        if f"client_proj_{client_idx}" in checkpoint:
            model.proj_heads.load_state_dict(checkpoint[f"client_proj_{client_idx}"])
        if f"optim_{client_idx}" in checkpoint:
            task_optimizers[client_idx].load_state_dict(checkpoint[f"optim_{client_idx}"])
        if f"proj_optim_{client_idx}" in checkpoint:
            projection_optimizers[client_idx].load_state_dict(checkpoint[f"proj_optim_{client_idx}"])
        if f"sched_{client_idx}" in checkpoint:
            task_schedulers[client_idx].load_state_dict(checkpoint[f"sched_{client_idx}"])
        if f"proj_sched_{client_idx}" in checkpoint:
            projection_schedulers[client_idx].load_state_dict(checkpoint[f"proj_sched_{client_idx}"])
    return checkpoint


def main():
    available_datasets = ["camelyon17", "prostate", "dgdr", "DGDR"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="store_true", help="whether to log")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--sfe_lr", type=float, default=1e-4, help="learning rate for SFE projection heads")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.99, help="Adam beta2")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Adam weight decay")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--iters", type=int, default=100, help="communication rounds")
    parser.add_argument("--wk_iters", type=int, default=1, help="local epochs per communication round")
    parser.add_argument("--lambda_icw", type=float, default=1e-4, help="ICW balancing weight")
    parser.add_argument("--icw_groups", type=int, default=0, help="ICW group count; 0 uses client number")
    parser.add_argument("--sfe_steps", type=int, default=1, help="SFE optimization steps after each local epoch")
    parser.add_argument("--disable_icw", action="store_true", help="disable intra-client whitening")
    parser.add_argument("--disable_sfl", action="store_true", help="disable SFL reweighting")
    parser.add_argument("--disable_sfe", action="store_true", help="disable SFE alignment loss")
    parser.add_argument("--nonnegative_sfl_weights", action="store_true", help="clamp cosine weights to non-negative values")
    parser.add_argument("--use_amp_norm", action="store_true", help="optionally enable HarmoFL amplitude normalization")
    parser.add_argument("--data", type=str, choices=available_datasets, default="camelyon17", help="dataset")
    parser.add_argument("--image_size", type=int, default=0, help="override input image size")
    parser.add_argument("--unet_init_features", type=int, default=64, help="U-Net base channel count")
    parser.add_argument("--save_path", type=str, default="../checkpoint/", help="path to save the checkpoint")
    parser.add_argument("--test_path", type=str, default="../checkpoint/", help="path to saved model for testing")
    parser.add_argument("--resume", action="store_true", help="resume training from latest checkpoint")
    parser.add_argument("--gpu", type=int, default=0, help="gpu device number")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--test", action="store_true", help="test model")
    parser.add_argument("--imbalance", action="store_true", help="do not truncate train data to same length")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument("--pin_memory", action="store_true", help="pin DataLoader memory")

    args = parser.parse_args()
    args.log = True if args.log else False
    args.data = args.data.lower()

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.save_path = os.path.join("../checkpoint", args.data, f"seed{seed}", "RRSFL_exp")
    os.makedirs(args.save_path, exist_ok=True)
    save_path = os.path.join(args.save_path, "RRSFL")

    server_model, loss_fun, datasets, _, _, train_loaders, val_loaders, test_loaders = initialize(args)

    print("# Device:", device)
    print("# Training Clients:{}".format(datasets))
    print("# ICW groups:", args.icw_groups)

    logfile = None
    if args.log:
        log_path = args.save_path.replace("checkpoint", "log")
        os.makedirs(log_path, exist_ok=True)
        logfile = open(os.path.join(log_path, "RRSFL.log"), "a")
        logfile.write("==={}===\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logfile.write("===Setting===\n")
        for k in list(vars(args).keys()):
            logfile.write("{}: {}\n".format(k, vars(args)[k]))

    client_num = len(datasets)
    models = [copy.deepcopy(server_model) for _ in range(client_num)]

    task_optimizers, projection_optimizers, task_schedulers, projection_schedulers = build_optimizers(args, models)

    if args.test:
        print("Loading snapshots...")
        checkpoint = torch.load(args.test_path, map_location=device)
        server_model.load_state_dict(checkpoint["server_model"], strict=False)
        for idx, test_loader in enumerate(test_loaders):
            test_loss, test_acc = test(args, server_model, test_loader, loss_fun, device)
            print("[Client-{}]  Test  Loss: {:.4f}, Test  Acc: {:.4f}".format(datasets[idx], test_loss, test_acc))
        return

    if args.resume:
        checkpoint = load_checkpoint(
            save_path + "_latest",
            server_model,
            models,
            task_optimizers,
            projection_optimizers,
            task_schedulers,
            projection_schedulers,
            device,
        )
        best_epoch, best_acc = checkpoint["best_epoch"], checkpoint["best_acc"]
        start_iter = int(checkpoint["a_iter"]) + 1
        print(f"Last time best:{best_epoch} acc :{best_acc}")
        print("Resume training from epoch {}".format(start_iter))
    else:
        best_epoch = 0
        best_acc = [0.0 for _ in range(client_num)]
        start_iter = 0

    for a_iter in range(start_iter, args.iters):
        for wi in range(args.wk_iters):
            epoch = wi + a_iter * args.wk_iters
            print("============ Train epoch {} ============".format(epoch))
            if logfile:
                logfile.write("============ Train epoch {} ============\n".format(epoch))

            for client_idx, model in enumerate(models):
                train_loss, task_loss, icw_loss, train_acc = train_one_client(
                    args,
                    model,
                    train_loaders[client_idx],
                    task_optimizers[client_idx],
                    loss_fun,
                    device,
                )
                task_schedulers[client_idx].step()
                print(
                    " Site-{:<10s}| Train Loss: {:.4f} | Task: {:.4f} | ICW: {:.4f} | Train Acc: {:.4f}".format(
                        datasets[client_idx], train_loss, task_loss, icw_loss, train_acc
                    )
                )
                if logfile:
                    logfile.write(
                        " Site-{:<10s}| Train Loss: {:.4f} | Task: {:.4f} | ICW: {:.4f} | Train Acc: {:.4f}\n".format(
                            datasets[client_idx], train_loss, task_loss, icw_loss, train_acc
                        )
                    )

        client_embeddings, sfe_loss = optimize_sfe_and_get_embeddings(
            args,
            models,
            train_loaders,
            projection_optimizers,
            device,
        )
        if not args.disable_sfe and args.sfe_steps > 0:
            for scheduler in projection_schedulers:
                scheduler.step()
        server_model, models, stage_weights = communication(args, server_model, models, client_embeddings, device)

        print(" SFE Loss: {:.6f}".format(sfe_loss))
        for stage_idx, weights in enumerate(stage_weights):
            print(" Stage-{} weights: {}".format(stage_idx + 1, ["{:.4f}".format(w) for w in weights]))
        if logfile:
            logfile.write(" SFE Loss: {:.6f}\n".format(sfe_loss))
            for stage_idx, weights in enumerate(stage_weights):
                logfile.write(" Stage-{} weights: {}\n".format(stage_idx + 1, weights))

        val_acc_list = [None for _ in range(client_num)]
        print("============== {} ==============".format("Global Validation"))
        if logfile:
            logfile.write("============== {} ==============\n".format("Global Validation"))
        for client_idx, _ in enumerate(models):
            val_loss, val_acc = test(args, server_model, val_loaders[client_idx], loss_fun, device)
            val_acc_list[client_idx] = val_acc
            print(" Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}".format(datasets[client_idx], val_loss, val_acc))
            if logfile:
                logfile.write(" Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}\n".format(datasets[client_idx], val_loss, val_acc))
                logfile.flush()

        print("============== {} ==============".format("Test"))
        if logfile:
            logfile.write("============== {} ==============\n".format("Test"))
        for client_idx, datasite in enumerate(datasets):
            _, test_acc = test(args, server_model, test_loaders[client_idx], loss_fun, device)
            print(" Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}".format(datasite, a_iter, test_acc))
            if logfile:
                logfile.write(" Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n".format(datasite, a_iter, test_acc))

        best_changed = False
        if np.mean(val_acc_list) > np.mean(best_acc):
            for client_idx in range(client_num):
                best_acc[client_idx] = val_acc_list[client_idx]
            best_epoch = a_iter
            best_changed = True
            print(" Best Epoch:{}".format(best_epoch))
            if logfile:
                logfile.write(" Best Epoch:{}\n".format(best_epoch))

        print(" Saving the latest checkpoint to {}...".format(save_path))
        if logfile:
            logfile.write(" Saving the latest checkpoint to {}...\n".format(save_path))
        if best_changed:
            save_checkpoint(
                save_path,
                server_model,
                models,
                task_optimizers,
                projection_optimizers,
                task_schedulers,
                projection_schedulers,
                best_epoch,
                best_acc,
                a_iter,
            )
        save_checkpoint(
            save_path + "_latest",
            server_model,
            models,
            task_optimizers,
            projection_optimizers,
            task_schedulers,
            projection_schedulers,
            best_epoch,
            best_acc,
            a_iter,
        )

    if logfile:
        logfile.flush()
        logfile.close()


if __name__ == "__main__":
    main()
