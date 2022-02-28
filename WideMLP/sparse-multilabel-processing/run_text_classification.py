"""
File: run_text_classification.py
Author: ANONYMIZED
Email: ANONYMIZED
Github: ANONYMIZED
Description: Run text classification experiments on TextGCN's datasets
"""

import csv
import logging

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

import scipy.sparse as sp
import torch
import matplotlib.pyplot as plt
from joblib import Memory
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import (AdamW, AutoTokenizer, get_linear_schedule_with_warmup)

from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

from tokenization import build_tokenizer_for_word_embeddings
from data import load_data, load_word_vectors
from models import MLP, collate_for_mlp

from multilabel_data import multilabel_collate_for_mlp, MultilabelDataset

try:
    import wandb

    WANDB = True
except ImportError:
    print("WandB not installed, to track experiments: pip install wandb")
    WANDB = False

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)
USE_CUDA = torch.cuda.is_available()
CACHE_DIR = 'cache/textclf'
MEMORY = Memory(CACHE_DIR, verbose=2)

VALID_DATASETS = ['reuters', 'dbpedia', 'goemotions', 'econbiz', 'pubmed', 'amazon', 'rcv1-v2', 'nyt']


def inverse_document_frequency(encoded_docs, vocab_size):
    """ Returns IDF scores in shape [vocab_size] """
    num_docs = len(encoded_docs)
    counts = sp.dok_matrix((num_docs, vocab_size))
    for i, doc in tqdm(enumerate(encoded_docs), desc="Computing IDF"):
        for j in doc:
            counts[i, j] += 1

    tfidf = TfidfTransformer(use_idf=True, smooth_idf=True)

    tfidf.fit(counts)

    return torch.FloatTensor(tfidf.idf_)


def pad(seqs, with_token=0, to_length=None):
    if to_length is None:
        to_length = max(len(seq) for seq in seqs)
    return [seq + (to_length - len(seq)) * [with_token] for seq in seqs]


def get_collate_for_transformer(pad_token_id):
    """ Closure to include padding in collate function """

    def _collate_for_transformer(examples):
        docs, labels = list(zip(*examples))
        input_ids = torch.tensor(pad(docs, with_token=pad_token_id), dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        attention_mask[input_ids == pad_token_id] = 0
        labels = torch.tensor(labels)
        token_type_ids = torch.zeros_like(input_ids)
        return input_ids, attention_mask, token_type_ids, labels

    return _collate_for_transformer


def train(args, train_data, valid_data, model, tokenizer):
    collate_fn = multilabel_collate_for_mlp
    train_loader = torch.utils.data.DataLoader(train_data,
                                               collate_fn=collate_fn,
                                               shuffle=True,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               pin_memory=('cuda' in str(args.device)))

    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               collate_fn=collate_fn,
                                               shuffle=True,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               pin_memory=('cuda' in str(args.device)))

    # len(train_loader) no of batches
    t_total = len(train_loader) // args.gradient_accumulation_steps * args.epochs

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    writer = SummaryWriter()

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_data))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Batch size  = %d", args.batch_size)
    logger.info("  Total train batch size (w. accumulation) = %d",
                args.batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    loss_vals = []
    tr_loss, logging_loss, nb_val_steps, vl_loss = 0.0, 0.0, 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(args.epochs, desc="Epoch")
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_loader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            outputs = model(batch[0], batch[1], batch[2])
            loss = outputs[0]
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                if WANDB:
                    wandb.log({'epoch': epoch,
                               'lr': scheduler.get_last_lr()[0],
                               'loss': loss})

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                avg_loss = (tr_loss - logging_loss) / args.logging_steps
                writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                writer.add_scalar('loss', avg_loss, global_step)
                logging_loss = tr_loss

        epoch_iterator_validation = tqdm(valid_loader, desc="Validating")
        for step, batch in enumerate(epoch_iterator_validation):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                outputs = model(batch[0], batch[1], batch[2])

            nb_val_steps += 1
            loss, logits = outputs[:2]
            vl_loss += loss.mean().item()

        vl_loss /= nb_val_steps
        loss_vals.append(vl_loss)
        if WANDB:
            wandb.log({'val_loss': vl_loss})

    writer.close()
    loss_plot(np.linspace(1, args.epochs, args.epochs).astype(int), loss_vals)
    return global_step, tr_loss / global_step


def evaluate(args, dev_or_test_data, model, tokenizer):
    N = len(dev_or_test_data)
    collate_fn = multilabel_collate_for_mlp
    data_loader = torch.utils.data.DataLoader(dev_or_test_data,
                                              collate_fn=collate_fn,
                                              num_workers=args.num_workers,
                                              batch_size=args.test_batch_size,
                                              pin_memory=('cuda' in str(args.device)),
                                              shuffle=False)
    all_logits = []
    all_targets = []
    nb_eval_steps, eval_loss = 0, 0.0

    acc = 0.0
    f1_samples = 0.0
    for batch in tqdm(data_loader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():

            # batch consist of (flat_inputs, lengths, labels)
            outputs = model(batch[0].to(torch.long), batch[1].to(torch.long), batch[2].to(torch.float))
            targets = batch[2].detach().cpu().numpy()

            # outputs [:2] should hold loss, logits
            nb_eval_steps += 1
            loss, logits = outputs[:2]
            eval_loss += loss.mean().item()

            # Thresholding
            logits = torch.sigmoid(logits.detach())
            logits = logits.cpu().numpy()
            logits[logits >= args.threshold] = 1
            logits[logits < args.threshold] = 0
            preds = logits

        N_batch = batch[2].size(0)  # Real batch size
        batch_weight = N_batch / N  # Compute average incrementally
        # f1 micro and macro will *not* work like this
        f1_samples += batch_weight * f1_score(targets, preds,
                                              average='samples')
        acc += batch_weight * accuracy_score(targets, preds)

    # f1_micro = f1_score(targets, preds, average='micro')
    # f1_macro = f1_score(targets, preds, average='macro')
    f1_micro, f1_macro = np.NaN, np.NaN  # compatibility

    eval_loss /= nb_eval_steps
    if WANDB:
        wandb.log({"test/acc": acc, "test/loss": eval_loss,
                   "test/f1_samples": f1_samples,
                   "test/f1_micro": f1_micro,
                   "test/f1_macro": f1_macro})
    return acc, eval_loss, f1_micro, f1_samples, f1_macro


def run_xy_model(args):
    print("Loading data...")

    if args.model_type == "mlp" and args.model_name_or_path is not None:
        print("Assuming to use word embeddings as both model_type=mlp and model_name_or_path are given")
        print("Using word embeddings -> forcing wordlevel tokenizer")
        vocab, embedding = load_word_vectors(args.model_name_or_path, unk_token="[UNK]")
        tokenizer = build_tokenizer_for_word_embeddings(vocab)
    else:
        tokenizer_name = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        embedding = None
    print("Using tokenizer:", tokenizer)

    do_truncate = not (args.stats_and_exit or args.model_type == 'mlp')
    if args.stats_and_exit:
        # We only compute dataset stats including length, so NOT truncate
        max_length = None
    elif args.model_type == 'mlp':
        max_length = None
    else:
        max_length = 512  # should hold for all used transformer models?

    enc_docs, enc_labels, train_mask, test_mask, label2index = load_data(args.dataset,
                                                                         tokenizer,
                                                                         args.dataset_folder,
                                                                         max_length=max_length,
                                                                         construct_textgraph=False,
                                                                         n_jobs=args.num_workers)
    print("Done")

    lens = np.array([len(doc) for doc in enc_docs])
    print("Min/max document length:", (lens.min(), lens.max()))
    print("Mean document length: {:.4f} ({:.4f})".format(lens.mean(), lens.std()))
    # enc_docs_arr, enc_labels_arr = np.array(enc_docs, dtype='object'), np.array(enc_labels)
    enc_docs_arr = np.array(enc_docs, dtype=object)
    enc_labels_arr = enc_labels  # is already csr matrix

    # train_data = list(zip(enc_docs_arr[train_mask], enc_labels_arr[train_mask, :]))
    # test_data = list(zip(enc_docs_arr[test_mask], enc_labels_arr[test_mask, :]))

    train_data = MultilabelDataset(enc_docs_arr[train_mask],
                                   enc_labels_arr[train_mask])
    test_data = MultilabelDataset(enc_docs_arr[test_mask],
                                  enc_labels_arr[test_mask])



    train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=200)

    print("N", len(enc_docs))
    print("N train", len(train_data))
    print("N valid", len(valid_data))
    print("N test", len(test_data))
    print("N classes", len(label2index))

    if args.stats_and_exit:
        print(
            "Warning: length stats depend on tokenizer and max_length of model, chose MLP to avoid trimming before computing stats.")
        exit(0)

    print("Initializing MLP")

    if embedding is not None:
        # Vocab size given by embedding
        vocab_size = None
    else:
        vocab_size = tokenizer.vocab_size

        if args.bow_aggregation == 'tfidf':
            print("Using IDF")
            idf = inverse_document_frequency(enc_docs_arr[train_mask], tokenizer.vocab_size).to(args.device)
        else:
            idf = None

        model = MLP(vocab_size, len(label2index),
                    num_hidden_layers=args.mlp_num_layers,
                    hidden_size=args.mlp_hidden_size,
                    embedding_dropout=args.mlp_embedding_dropout,
                    dropout=args.mlp_dropout,
                    mode=args.bow_aggregation,
                    pretrained_embedding=embedding,
                    idf=idf,
                    freeze=args.freeze_embedding)

    model.to(args.device)

    if WANDB:
        wandb.watch(model, log_freq=args.logging_steps)

    train(args, train_data, valid_data, model, tokenizer)
    acc, eval_loss, f1_micro, f1_samples, f1_macro = evaluate(args, test_data, model, tokenizer)
    print(f"[{args.dataset}] Test accuracy: {acc:.4f}, Eval loss: {eval_loss}")
    return acc, eval_loss, f1_micro, f1_samples, f1_macro


def loss_plot(epochs, loss):
    plt.plot(epochs, loss, color='red', label='loss')
    plt.xlabel("epochs")
    plt.title("validation loss")
    plt.savefig("val_loss.png")


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', default='reuters', choices=VALID_DATASETS)
    parser.add_argument("--dataset_folder", default="data/multi_label_datasets", type=str,
                        help="Path to dataset folder")
    parser.add_argument("--model_type", default="mlp", type=str,
                        help="Model type: either 'mlp' or 'distilbert'",
                        choices=["mlp"])
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="Optional path to word embedding with model type 'mlp' OR huggingface shortcut name such as distilbert-base-uncased for model type 'distilbert'")
    parser.add_argument("--results_file", default="results_mlp.csv",
                        help="Store results to this results file")

    ## Training config
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Label threshold. Default: 0.5")
    parser.add_argument("--test_batch_size", type=int, default=None,
                        help="Batch size for testing (defaults to train batch size)")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")

    parser.add_argument('--unfreeze_embedding', dest="freeze_embedding", default=True,
                        action='store_false', help="Allow updating pretrained embeddings")

    ## Training Hyperparameters
    parser.add_argument("--learning_rate", default=0.001, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    ## Other parameters
    parser.add_argument("--tokenizer_name", default="bert-base-uncased", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--num_workers", default=4, type=int,
                        help="Number of workers")

    parser.add_argument("--stats_and_exit", default=False,
                        action='store_true',
                        help="Print dataset stats and exit.")

    # MLP Params
    parser.add_argument("--mlp_num_layers", default=1, type=int, help="Number of hidden layers within MLP")
    parser.add_argument("--mlp_hidden_size", default=1024, type=int, help="Hidden dimension for MLP")
    parser.add_argument("--bow_aggregation", default="mean", choices=["mean", "sum", "tfidf"],
                        help="Aggregation for bag-of-words models (such as MLP)")
    parser.add_argument("--mlp_embedding_dropout", default=0.5, type=float,
                        help="Dropout for embedding / first hidden layer ")
    parser.add_argument("--mlp_dropout", default=0.5, type=float, help="Dropout for all subsequent layers")

    parser.add_argument("--comment", help="Some comment for the experiment")
    args = parser.parse_args()

    if args.model_type in ['mlp']:
        assert args.tokenizer_name or args.model_name_or_path, "Please supply tokenizer for MLP via --tokenizer_name or provide an embedding via --model_name_or_path"
    else:
        assert args.model_name_or_path, f"Please supply --model_name_or_path for {args.model_type}"

    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.test_batch_size = args.batch_size if args.test_batch_size is None else args.test_batch_size

    if WANDB:
        wandb.init(project="text-clf")
        wandb.config.update(args)

    acc, eval_loss, f1_micro, f1_samples, f1_macro = {
        'mlp': run_xy_model
    }[args.model_type](args)
    if args.results_file:
        file_exists = os.path.isfile(args.results_file)
        with open(args.results_file, 'a', newline='') as csvfile:
            headers = ['Model', 'Dataset', 'Epochs', 'Batch', 'Learning rate', 'eval loss', 'acc', 'f1 samples',
                       'f1 micro', 'f1 macro']
            csv_writer = csv.writer(csvfile)
            if not file_exists:
                csv_writer.writerow(headers)
            csv_writer.writerow(
                [args.model_type, args.dataset, args.epochs, args.batch_size, args.learning_rate, eval_loss, acc,
                 f1_samples, f1_micro, f1_macro])


if __name__ == '__main__':
    main()
