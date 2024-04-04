import torch
from piqa import PSNR


def f1(y_pred, y):
    tp = torch.sum((y_pred > 0.5) * y)
    fp = torch.sum((y_pred > 0.5) * (1 - y))
    fn = torch.sum(((1 - y_pred) > 0.5) * y)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return 2 * (precision * recall) / (precision + recall)


class InverseCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(InverseCrossEntropyLoss, self).__init__()

    def forward(self, y_pred, y):
        return -torch.nn.functional.cross_entropy(y_pred, y)


def evaluate(task1, task2, fold, model, test_loader, device, logger):
    if logger.type_training == "pretext":
        if task1 == "j" or task1 == "p":
            metric_fn = InverseCrossEntropyLoss()
        else:
            metric_fn = PSNR()
    elif logger.type_training == "downstream":
        metric_fn = f1

    model.eval()
    total_metric = 0
    with torch.no_grad():
        for i, (sample, gt) in enumerate(test_loader):
            sample = sample.to(device)
            gt = gt.to(device)

            outputs = model(sample)
            metric = metric_fn(outputs, gt)
            total_metric += metric.item()
            print(f"Batch {i} metric: {metric.item()}")
    avg_metric = total_metric / len(test_loader)
    print(f"Average metric: {avg_metric}")
    logger.log_evaluation(task1, task2, fold, avg_metric)
