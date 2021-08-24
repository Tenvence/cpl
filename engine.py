import sklearn.metrics as metrics
import torch
import torch.cuda.amp as amp


def train(model, optimizer, lr_scheduler, data_loader):
    model.train()
    scaler = amp.GradScaler()
    losses = []

    for img, label in data_loader:
        img = img.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        optimizer.zero_grad()
        with amp.autocast():
            loss = model(img, label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        losses.append(loss.detach())

    return sum(losses) / len(losses)


@torch.no_grad()
def val(model, data_loader):
    model.eval()
    pred_list, gt_list = [], []
    for img, label in data_loader:
        img = img.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        with amp.autocast():
            assign_distribution = model(img)
            pred = torch.argmax(assign_distribution, dim=-1)

        pred_list.append(pred)
        gt_list.append(label)

    pred_list = torch.cat(pred_list, dim=0)
    gt_list = torch.cat(gt_list, dim=0)

    acc = metrics.accuracy_score(gt_list.cpu(), pred_list.cpu()) * 100
    mae = metrics.mean_absolute_error(gt_list.cpu(), pred_list.cpu())

    acc = torch.tensor(acc).cuda()
    mae = torch.tensor(mae).cuda()

    return acc, mae
