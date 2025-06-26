import torch

def forward_scaled_loss(model, loss_function, batch, loss_modifier):
    for i in batch.shape[1]:
        logits = model(batch[:, :i]).logits[:, -1]
        labels = batch[:, i]
        loss = (loss_function(logits, labels) * loss_modifier[i]).mean()
        loss.backward()

batch = torch.tensor([[1,2,3], [4,5,6]])
loss_modifier = torch.tensor([[0.05,0.1,0.5], [1.0,0.1,0.05]])
