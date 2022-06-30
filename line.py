from train_line import train, test, model, train_loader, test_loader, loss_function
import torch

epochs = 501

for epoch in range(epochs):
    train_loss = train(model, train_loader, loss_function=loss_function)
    if epoch % 100 == 0:
        print(f"epoch: {epoch} | loss: #{train_loss}")

test_loss = test(model, test_loader, loss_function=loss_function)
print(f"test loss: {test_loss}")

torch.save(model.state_dict(), "./models/line_model.pth")
