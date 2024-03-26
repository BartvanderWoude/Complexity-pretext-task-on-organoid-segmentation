import torch


def selfprediction_train(model, train_loader, optimizer, loss_fn, epochs, device):
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
            torch.save(model.state_dict(), 'output/model.pth')
