import torch

from piqa import PSNR


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
