val = torch.tensor(batch['input_ids'], dtype=torch.long).to(device)
logits = model(val).logits
ttl_loss = torch.zeros(1,dtype=torch.float32).to(device)
for i in full_range:
    loss = loss_function(logits[:, i - 1], val[:, i])
    if i > 26 or i < 6:
        losses.append(loss.item())
    else:
        loss = loss * 0.1
    ttl_loss += loss
ttl_loss.backward()
optimizer.step()
optimizer.zero_grad()
lr_scheduler.step()
