original_train = Dataset.from_file("./train.arrow")
test = Dataset.from_file("./test.arrow")
tokenizer = GPT2TokenizerFast.from_pretrained('./llama-it-2')
device = 'cuda'
config = AutoConfig.from_pretrained('./basis_for_train')
#model = AutoModelForCausalLM.from_pretrained('./basis_for_train').to(device)
model = AutoModelForCausalLM.from_config(config).to(device)

full_range = range(1, 128)
losses = []
#optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005)
optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=2e-4)
optimizer.train()
lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
loss_function = torch.nn.CrossEntropyLoss(reduction='none')
loss_count = 0
min_loss = 100
batch_sizes = [16, 8]
ttl_sub_epochs = 0
ttl_count = 1
x_token = torch.tensor(tokenizer.encode("x")).to(device)
xx_token = torch.tensor(tokenizer.encode("xx")).to(device)
xxx_token = torch.tensor(tokenizer.encode("xxx")).to(device)
min_xxxx_token = torch.tensor(tokenizer.encode("xxxx")).to(device)

for batch_size in batch_sizes:
    train = original_train.batch(batch_size=batch_size).shuffle()
    sub_epoch = 50000 // batch_size
    for val in train:
        val = torch.tensor(val['input_ids'], dtype=torch.long).to(device)
        logits = model(val).logits
        ttl_loss = torch.zeros(1, dtype=torch.float32).to(device)
        current = val[:, 0]
        _ = val[:, 1]
        last = False
        last_last = False
        current_ = -1
        last_ = -1
        last_last_ = -1
        for i in full_range:
            last_last = last
            last = current
            current = val[:, i]
            loss = loss_function(logits[:, i - 1], current)
            losses.append(loss.mean().item())
            mod = 1 + i / 64
            if i < 6:
                pass

            elif i > 26:

                if i > 28:
                    last_last_ = last_
                if i > 27:
                    last_ = current_
                current_ = torch.argwhere(current == _).squeeze(1)
                modifier = torch.full([batch_size], 1).to(device)
                if i > 27 and last_.shape[0]:
                    x = torch.argwhere(current[last_] == x_token).squeeze(1)
                    if x.shape[0]:
                        modifier[last_[x]] += 2
                    xx = torch.argwhere(current[last_] == xx_token).squeeze(1)
                    if xx.shape[0]:
                        modifier[last_[xx]] += 2
                    xxx = torch.argwhere(current[last_] == xxx_token).squeeze(1)
                    if xxx.shape[0]:
                        modifier[last_[xxx]] += 2
                if i > 28 and last_last_.shape[0]:
                    xxxx = torch.argwhere(current[last_last_] >= min_xxxx_token).squeeze(1)
                    if xxxx.shape[0]:
                        modifier[last_last_[xxxx]] += 2
                if current_.shape[0]:
                    modifier[current_] += 1

                loss = loss * mod * modifier
            else:
                loss = loss * 0.5
            ttl_loss += loss.mean()
        ttl_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
