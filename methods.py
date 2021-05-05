from models import *
from tqdm import tqdm

def train(model, iterator, optimizer, criterion, clip, device):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(tqdm(iterator)):
        src = batch[0].to(device)
        trg = batch[1].to(device)

        optimizer.zero_grad()

        output = model(src, trg)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch[0].to(device)
            trg = batch[1].to(device)
            output = model(src, trg, teacher_forcing_ratio=0)  # turn off teacher forcing

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]
            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def predict(model, loader, device):
    outputs = []
    model.eval()

    with torch.no_grad():
        for x, batch in enumerate(tqdm(loader)):
            src = batch[0].unsqueeze(dim=0).to(device)
            logits = model(src, src, teacher_forcing_ratio=0)
            outputs.append(logits)
    return outputs


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def decode_prediction_beam(pred, vocab):
    predicted_sent = []
    for i, p in enumerate(pred):
        predicted_sent.append(vocab[int(p)])

    string = ''
    word = predicted_sent[0]
    count = 1
    while word != '<eos>' and count < len(predicted_sent):
        string += word + ' '
        word = predicted_sent[count]
        count += 1

    string = string[:-1]

    return string
