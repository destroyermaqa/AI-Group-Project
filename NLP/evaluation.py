import torch
import torch.nn as nn
import math
from collections import Counter

from pipeline import get_data_pipeline
from model_withoutAttention import Encoder as VanillaEncoder, VanillaDecoder, Seq2Seq as VanillaSeq2Seq
from model_withAttention import Encoder as AttentionEncoder, AttentionDecoder, Seq2Seq as AttentionSeq2Seq


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def get_ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def bleu_score(hypotheses, references, max_n=4):
    """Corpus-level BLEU with brevity penalty."""
    clip_counts = Counter()
    total_counts = Counter()
    hyp_len = 0
    ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_len += len(hyp)
        ref_len += len(ref)
        for n in range(1, max_n + 1):
            hyp_ngrams = Counter(get_ngrams(hyp, n))
            ref_ngrams = Counter(get_ngrams(ref, n))
            for ngram, count in hyp_ngrams.items():
                clip_counts[n] += min(count, ref_ngrams[ngram])
            total_counts[n] += len(get_ngrams(hyp, n))

    precisions = []
    for n in range(1, max_n + 1):
        if total_counts[n] == 0:
            precisions.append(0.0)
        else:
            precisions.append(clip_counts[n] / total_counts[n])

    if min(precisions) == 0:
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / max_n
    bp = min(1.0, math.exp(1 - ref_len / hyp_len)) if hyp_len > 0 else 0.0
    return bp * math.exp(log_avg) * 100


def rouge_l(hypothesis, reference):
    """Sentence-level ROUGE-L F1 via LCS."""
    n, m = len(hypothesis), len(reference)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if hypothesis[i-1] == reference[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[n][m]
    precision = lcs / n if n > 0 else 0.0
    recall    = lcs / m if m > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return f1


def corpus_rouge_l(hypotheses, references):
    scores = [rouge_l(h, r) for h, r in zip(hypotheses, references)]
    return sum(scores) / len(scores) * 100 if scores else 0.0


def exact_match(hypotheses, references):
    matches = sum(1 for h, r in zip(hypotheses, references) if h == r)
    return matches / len(hypotheses) * 100 if hypotheses else 0.0


def token_accuracy(hypotheses, references):
    correct = total = 0
    for h, r in zip(hypotheses, references):
        length = min(len(h), len(r))
        correct += sum(1 for i in range(length) if h[i] == r[i])
        total += max(len(h), len(r))      # penalise length mismatches
    return correct / total * 100 if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Greedy decode helper
# ---------------------------------------------------------------------------

def decode_batch(model, src, trg, device):
    """Run the model without teacher forcing and return token sequences."""
    model.eval()
    with torch.no_grad():
        output = model(src, trg, teacher_forcing_ratio=0.0)   # [B, T, vocab]
    predicted_ids = output.argmax(-1)                          # [B, T]
    return predicted_ids


def ids_to_tokens(id_tensor, vocab, eos_token="<EOS>", pad_token="<PAD>", sos_token="<SOS>"):
    """Convert a 1-D id tensor to a list of token strings, stopping at EOS."""
    eos_idx = vocab.stoi.get(eos_token, None)
    pad_idx = vocab.stoi.get(pad_token, None)
    sos_idx = vocab.stoi.get(sos_token, None)
    tokens = []
    for idx in id_tensor.tolist():
        if idx == eos_idx:
            break
        if idx in (pad_idx, sos_idx):
            continue
        tokens.append(vocab.itos[idx])
    return tokens


# ---------------------------------------------------------------------------
# Full evaluation loop
# ---------------------------------------------------------------------------

def evaluate_model(model, loader, de_vocab, device, model_name):
    print(f"\n{'='*55}")
    print(f"  Evaluating: {model_name}")
    print(f"{'='*55}")

    hypotheses, references = [], []

    model.eval()
    with torch.no_grad():
        for src, trg in loader:
            src, trg = src.to(device), trg.to(device)
            predicted = decode_batch(model, src, trg, device)   # [B, T]

            for i in range(trg.size(0)):
                hyp = ids_to_tokens(predicted[i, 1:], de_vocab)   # skip <SOS>
                ref = ids_to_tokens(trg[i, 1:], de_vocab)
                hypotheses.append(hyp)
                references.append(ref)

    bleu   = bleu_score(hypotheses, references)
    rouge  = corpus_rouge_l(hypotheses, references)
    em     = exact_match(hypotheses, references)
    tok_acc = token_accuracy(hypotheses, references)

    print(f"  BLEU Score        : {bleu:.2f}")
    print(f"  ROUGE-L F1        : {rouge:.2f}")
    print(f"  Exact Match       : {em:.2f}%")
    print(f"  Token Accuracy    : {tok_acc:.2f}%")
    print(f"{'='*55}\n")

    return {"bleu": bleu, "rouge_l": rouge, "exact_match": em, "token_accuracy": tok_acc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    BATCH_SIZE = 32
    loader, en_vocab, de_vocab = get_data_pipeline(batch_size=BATCH_SIZE)

    INPUT_DIM  = len(en_vocab)
    OUTPUT_DIM = len(de_vocab)
    ENC_EMB_DIM = DEC_EMB_DIM = 256
    HIDDEN_DIM  = 512
    NUM_LAYERS  = 1
    DROPOUT     = 0.5

    # ── Vanilla model ────────────────────────────────────────────────────────
    vanilla_enc = VanillaEncoder(INPUT_DIM, ENC_EMB_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    vanilla_dec = VanillaDecoder(OUTPUT_DIM, DEC_EMB_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    vanilla_model = VanillaSeq2Seq(vanilla_enc, vanilla_dec, device).to(device)
    vanilla_model.load_state_dict(torch.load('vanilla_model_best.pt', map_location=device))

    # ── Attention model ──────────────────────────────────────────────────────
    attn_enc = AttentionEncoder(INPUT_DIM, ENC_EMB_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    attn_dec = AttentionDecoder(OUTPUT_DIM, DEC_EMB_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    attn_model = AttentionSeq2Seq(attn_enc, attn_dec, device).to(device)
    attn_model.load_state_dict(torch.load('attention_model_best.pt', map_location=device))

    # ── Run evaluation ───────────────────────────────────────────────────────
    vanilla_results = evaluate_model(vanilla_model, loader, de_vocab, device, "Vanilla Seq2Seq (No Attention)")
    attn_results    = evaluate_model(attn_model,    loader, de_vocab, device, "Seq2Seq + Bahdanau Attention")

    # ── Side-by-side comparison ──────────────────────────────────────────────
    print("\n" + "="*55)
    print("  COMPARISON SUMMARY")
    print("="*55)
    print(f"  {'Metric':<22} {'Vanilla':>10} {'Attention':>10}")
    print("-"*55)
    metrics = [
        ("BLEU Score",     "bleu"),
        ("ROUGE-L F1",     "rouge_l"),
        ("Exact Match (%)", "exact_match"),
        ("Token Acc. (%)",  "token_accuracy"),
    ]
    for label, key in metrics:
        v = vanilla_results[key]
        a = attn_results[key]
        winner = "<" if a > v else (">" if v > a else "=")
        print(f"  {label:<22} {v:>10.2f} {a:>10.2f}  {winner}")
    print("="*55)
    print("  < = Attention wins   > = Vanilla wins   = = Tie")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
