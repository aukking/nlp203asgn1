import torch.utils.data

from methods import *
from data_processing import *

N_EPOCHS = 5
CLIP = 1
best_valid_loss = float("inf")

BATCH_SIZE = 128 
UNK_THRESH = 10
TRAIN = True
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

samples = -1
src_truncate = 400
train_data_sources = get_data('data/train.txt.src', samples, src_truncate)
train_data_targets = get_data('data/train.txt.tgt', samples)
word2idx, id2word = generate_vocab(train_data_sources, UNK_THRESH)

INPUT_DIM = len(word2idx)
OUTPUT_DIM = len(word2idx)
ENC_EMB_DIM = 32
DEC_EMB_DIM = 32
ENC_HID_DIM = 64 
DEC_HID_DIM = 64
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
BEAM_SIZE = 2
MAX_SEQ_LEN = 200
# the last item in the word2idx will be the pad token
PAD_IDX = 3
UNK_IDX = 0
SOS_IDX = 1
EOS_IDX = 2

train_dataset = TrainDataset(train_data_sources, train_data_targets, word2idx, UNK_IDX, SOS_IDX, EOS_IDX)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=lambda batch: prepare_batch(batch, PAD_IDX))

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

if TRAIN:
    model = Seq2Seq(enc, dec, PAD_IDX, device, SOS_IDX, EOS_IDX, beam_size=BEAM_SIZE, max_seq_len=MAX_SEQ_LEN).to(device)
    print("device:", device)
    print("parameters:", count_parameters(model))
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    for epoch in range(N_EPOCHS):

        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, CLIP, device)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}")
    torch.save(model.state_dict(), 'models/new_model2.pt')
else:
    # device = 'cpu'
    model = Seq2Seq(enc, dec, PAD_IDX, device, SOS_IDX, EOS_IDX, beam_size=BEAM_SIZE, max_seq_len=MAX_SEQ_LEN).to(
        device)
    print("device:", device)
    print("parameters:", count_parameters(model))
    model.load_state_dict(torch.load('models/new_model2.pt'))

    test_data_sources = get_data('data/test.txt.src', 1)
    test_dataset = TestDataset(test_data_sources, word2idx, UNK_IDX, SOS_IDX, EOS_IDX)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    preds = predict(model, test_loader, device)
    predictions = [decode_prediction_beam(p, id2word) for p in preds]
    print(test_data_sources[0])
    print(predictions[0])
