import torch

from piqa import PSNR


def f1(y_pred, y):
    tp = torch.sum((y_pred > 0.5) * y)
    fp = torch.sum((y_pred > 0.5) * (1 - y))
    fn = torch.sum(((1 - y_pred) > 0.5) * y)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return 2 * (precision * recall) / (precision + recall)


def evaluate_downstream(task1, task2, fold, model, test_loader, device, logger):
    model.eval()
    total_f1 = 0
    with torch.no_grad():
        for i, (sample, gt) in enumerate(test_loader):
            sample = sample.to(device)
            gt = gt.to(device)

            outputs = model(sample)
            f1_score = f1(outputs, gt)
            total_f1 += f1_score.item()
            print(f"Batch {i} F1: {f1_score.item()}")

    avg_f1 = total_f1 / len(test_loader)
    print(f"Average F1: {avg_f1}")
    logger.log_evaluation(task1, task2, fold, avg_f1)


def evaluate_selfprediction(task1, task2, fold, model, test_loader, device, logger):
    loss_fn = PSNR()

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, (sample, gt) in enumerate(test_loader):
            sample = sample.to(device)
            gt = gt.to(device)

            outputs = model(sample)
            loss = loss_fn(outputs, gt)
            total_loss += loss.item()
            print(f"Batch {i} PSNR: {loss.item()}")

    avg_loss = total_loss / len(test_loader)
    print(f"Average PSNR: {avg_loss}")
    logger.log_evaluation(task1, task2, fold, avg_loss)
