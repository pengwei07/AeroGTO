import sys

sys.path.append("/workspace/AeroGTO/paddle_project")
import os
import numpy as np
import paddle
from paddle_utils import *
from tqdm import tqdm


def get_l2_loss(output, target):
    error = output - target
    norm_error_sample = paddle.linalg.norm(x=error, axis=-2) / (
        paddle.linalg.norm(x=target, axis=-2) + 1e-06
    )
    if tuple(norm_error_sample.shape)[-1] == 1:
        norm_error_channnel = norm_error_sample.squeeze(axis=-1)
    else:
        norm_error_channnel = paddle.mean(x=norm_error_sample, axis=-1)
    norm_error_batch = paddle.mean(x=norm_error_channnel)
    return norm_error_batch


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = tuple(x.shape)[0]
        h = 1.0 / (tuple(x.shape)[1] - 1.0)
        all_norms = h ** (self.d / self.p) * np.linalg.norm(
            x.reshape((num_examples, -1)) - y.reshape((num_examples, -1)), self.p, 1
        )
        if self.reduction:
            if self.size_average:
                return np.mean(all_norms)
            else:
                return np.sum(all_norms)
        return all_norms

    def rel(self, x, y):
        diff_norms = np.linalg.norm(x - y, 2)
        y_norms = np.linalg.norm(y, self.p)
        if self.reduction:
            if self.size_average:
                return np.mean(diff_norms / y_norms)
            else:
                return np.sum(diff_norms / y_norms)
        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def get_val_loss(output_hat, target, if_rescale, info):
    device = output_hat.place
    target = target.to(device)
    info["mean"] = paddle.to_tensor(data=info["mean"]).to(device)
    info["std"] = paddle.to_tensor(data=info["std"]).to(device)
    losses = {}
    losses["L2_norm"] = get_l2_loss(
        output_hat, (target - info["mean"]) / info["std"]
    ).item()
    if if_rescale:
        output_hat = output_hat * info["std"] + info["mean"]
    loss_fn = LpLoss(size_average=True)
    target = target.squeeze(axis=0).detach().cpu().numpy()
    output_hat = output_hat.squeeze(axis=0).detach().cpu().numpy()
    losses["L2"] = loss_fn(output_hat, target)
    return losses


def get_train_loss(output_hat, target, loss_flag, if_rescale, info):
    device = output_hat.place
    target = target.to(device)
    info["mean"] = paddle.to_tensor(data=info["mean"]).to(device)
    info["std"] = paddle.to_tensor(data=info["std"]).to(device)
    losses = {}
    criterion = paddle.nn.MSELoss()
    if loss_flag == "L2_loss_norm":
        losses["loss"] = get_l2_loss(
            output_hat[..., :1], (target - info["mean"]) / info["std"]
        )
    elif loss_flag == "MSE_loss_norm":
        losses["loss"] = criterion(
            output_hat[..., :1], (target - info["mean"]) / info["std"]
        )
    losses["L2_norm"] = get_l2_loss(
        output_hat[..., :1], (target - info["mean"]) / info["std"]
    ).item()
    if if_rescale:
        output_hat = output_hat[..., :1] * info["std"] + info["mean"]
    else:
        output_hat = output_hat
    if loss_flag == "L2_loss":
        losses["loss"] = get_l2_loss(output_hat, target)
    elif loss_flag == "MSE_loss":
        losses["loss"] = criterion(output_hat, target)
    losses["L2"] = get_l2_loss(output_hat, target).item()
    return losses


def train(args, model, train_dataloader, optim, device):
    model.train()
    loss = 0
    L2 = 0
    L2_norm = 0
    num = 0
    for i, [input, t] in enumerate(tqdm(train_dataloader, desc="Training")):
        optim.clear_gradients(set_to_zero=False)
        gt = input["gt"]
        node_pos = input["node_pos"]
        edges = input["edges"]
        output_hat = model(node_pos.to(device), edges.to(device))
        if not args.task == "cd_pred":
            output_hat = output_hat * input["mask"].to(device)
        costs = get_train_loss(
            output_hat,
            gt,
            args.train["loss_flag"],
            args.train["if_rescale"],
            args.train["info"],
        )
        costs["loss"].backward()
        optim.step()
        loss = loss + costs["loss"].item()
        batch_num = tuple(gt.shape)[0]
        num = num + batch_num
        L2 = L2 + costs["L2"] * batch_num
        L2_norm = L2_norm + costs["L2_norm"] * batch_num
    batch_error = {}
    batch_error["loss"] = loss / num
    batch_error["L2"] = L2 / num
    batch_error["L2_norm"] = L2_norm / num
    return batch_error


def validate(args, model, val_dataloader, device):
    model.eval()
    L2 = 0
    L2_norm = 0
    num = 0
    with paddle.no_grad():
        for i, [input, t] in enumerate(tqdm(val_dataloader, desc="Validation")):
            gt = input["gt"]
            node_pos = input["node_pos"]
            edges = input["edges"]
            output_hat = model(node_pos.to(device), edges.to(device))
            if not args.task == "cd_pred":
                output_hat = output_hat * input["mask"].to(device)
            costs = get_val_loss(
                output_hat, gt, args.train["if_rescale"], args.train["info"]
            )
            batch_num = tuple(gt.shape)[0]
            num = num + batch_num
            L2 = L2 + costs["L2"] * batch_num
            L2_norm = L2_norm + costs["L2_norm"] * batch_num
    batch_error = {}
    batch_error["L2"] = L2 / num
    batch_error["L2_norm"] = L2_norm / num
    return batch_error


def infer(args, model, test_dataloader, device):
    model.eval()
    info = args.train["info"]
    info["mean"] = paddle.to_tensor(data=info["mean"]).to(device)
    info["std"] = paddle.to_tensor(data=info["std"]).to(device)
    with paddle.no_grad():
        for i, [input, name] in enumerate(test_dataloader):
            node_pos = input["node_pos"]
            edges = input["edges"]
            output_hat = model(node_pos.to(device), edges.to(device))
            if args.train["if_rescale"]:
                output_hat = output_hat[..., :1] * info["std"] + info["mean"]
            else:
                output_hat = output_hat
            model_name = args.model["name"]
            save_path = os.path.join(args.save_path, f"{model_name}_test_infer_result")
            os.makedirs(save_path, exist_ok=True)
            output_hat = output_hat.detach().cpu().numpy()
            np.save(f"{save_path}/{name[0]}.npy", output_hat.reshape(-1))
    print("Finished!")
