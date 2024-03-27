import torch


def selfprediction_train(model, train_loader, val_loader, optimizer, loss_fn, fold, epochs, device, logger):
    model.train()

    for epoch in range(epochs):
        for i, (sample, gt) in enumerate(train_loader):
            sample = sample.to(device)
            gt = gt.to(device)

            optimizer.zero_grad()
            pred = model(sample)
            loss = loss_fn(pred, gt)
            loss.backward()
            optimizer.step()

            print(f'Epoch {epoch} Iteration {i} Loss: {loss.item()}')
            logger.log_training_loss(fold, epoch, loss.item())
            if i % 10 == 0:
                logger.save_model(model, fold)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for i, (sample, gt) in enumerate(val_loader):
                sample = sample.to(device)
                gt = gt.to(device)

                pred = model(sample)
                loss = loss_fn(pred, gt)
                val_loss += loss.item()
            logger.log_validation_loss(fold, epoch, val_loss / len(val_loader))
