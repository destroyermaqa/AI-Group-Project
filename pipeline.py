import torch
import spacy
import requests
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

# 1. Tokenizatorları yükləyirik
try:
    spacy_eng = spacy.load("en_core_web_sm")
    spacy_ger = spacy.load("de_core_news_sm")
except OSError:
    print("Xəta: Spacy modelləri tapılmadı. Terminalda bunları icra edin:")
    print("python -m spacy download en_core_web_sm")
    print("python -m spacy download de_core_news_sm")

class Vocabulary:
    def __init__(self, freq_threshold):
        # Xüsusi tokenlər
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(str(text))]

    @staticmethod
    def tokenizer_ger(text):
        return [tok.text.lower() for tok in spacy_ger.tokenizer(str(text))]

    def build_vocabulary(self, sentence_list, lang="eng"):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            tokens = self.tokenizer_eng(sentence) if lang == "eng" else self.tokenizer_ger(sentence)
            for word in tokens:
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text, lang="eng"):
        tokens = self.tokenizer_eng(text) if lang == "eng" else self.tokenizer_ger(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokens]

class TranslationDataset(Dataset):
    def __init__(self, src_sentences, trg_sentences, src_vocab, trg_vocab):
        self.src_sentences = src_sentences
        self.trg_sentences = trg_sentences
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, index):
        src_text = self.src_sentences[index]
        trg_text = self.trg_sentences[index]

        # Rəqəmsallaşdırma və xüsusi tokenlərin əlavə edilməsi
        src_indices = [self.src_vocab.stoi["<SOS>"]] + self.src_vocab.numericalize(src_text, "eng") + [self.src_vocab.stoi["<EOS>"]]
        trg_indices = [self.trg_vocab.stoi["<SOS>"]] + self.trg_vocab.numericalize(trg_text, "ger") + [self.trg_vocab.stoi["<EOS>"]]

        return torch.tensor(src_indices), torch.tensor(trg_indices)

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        srcs = [item[0] for item in batch]
        trgs = [item[1] for item in batch]
        
        # Padding: Bütün cümlələri batch-dəki ən uzun cümləyə bərabər edirik
        srcs = pad_sequence(srcs, batch_first=True, padding_value=self.pad_idx)
        trgs = pad_sequence(trgs, batch_first=True, padding_value=self.pad_idx)
        
        return srcs, trgs

def get_data_pipeline(batch_size=32, max_len=15, freq_threshold=2):
    print("--- Məlumatlar GitHub-dan birbaşa endirilir ---")
    
    url_en = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/tok/train.lc.norm.tok.en"
    url_de = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/tok/train.lc.norm.tok.de"
    
    try:
        # Requests ilə datanı çəkirik
        en_data = requests.get(url_en).text.splitlines()
        de_data = requests.get(url_de).text.splitlines()
        
        # İlk 10000 cümləni götürək (test və təlim üçün kifayətdir)
        train_src_raw = en_data[:10000]
        train_trg_raw = de_data[:10000]
    except Exception as e:
        print(f"Bağlantı xətası: {e}. Kodu sınaq rejiminə keçiririk.")
        train_src_raw = ["two young white men are near some ice ."] * 200
        train_trg_raw = ["zwei junge weiße männer sind in der nähe von eis ."] * 200

    # 1. Uzunluğa görə filtrləmə (Tələb: 10-15 token)
    src_data, trg_data = [], []
    for s, t in zip(train_src_raw, train_trg_raw):
        if 0 < len(str(s).split()) <= max_len and 0 < len(str(t).split()) <= max_len:
            src_data.append(s)
            trg_data.append(t)

    print(f"Filtrlənmiş cümlə sayı: {len(src_data)}")

    # 2. Lüğətlərin qurulması
    src_vocab = Vocabulary(freq_threshold)
    src_vocab.build_vocabulary(src_data, lang="eng")
    
    trg_vocab = Vocabulary(freq_threshold)
    trg_vocab.build_vocabulary(trg_data, lang="ger")

    # 3. Dataset və DataLoader
    dataset = TranslationDataset(src_data, trg_data, src_vocab, trg_vocab)
    pad_idx = src_vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )

    return loader, src_vocab, trg_vocab

# --- Yoxlama (Yalnız bu faylı işə saldıqda işləyir) ---
if __name__ == "__main__":
    # Kitabxanaları yoxla: pip install requests
    loader, en_vocab, de_vocab = get_data_pipeline(batch_size=32)
    
    print(f"\nUğurlu! Sistem hazırdır.")
    print(f"İngilis lüğəti ölçüsü: {len(en_vocab)}")
    print(f"Alman lüğəti ölçüsü: {len(de_vocab)}")

    for src, trg in loader:
        print(f"Batch Tenser Ölçüsü (Giriş): {src.shape}")
        print(f"Batch Tenser Ölçüsü (Çıxış): {trg.shape}")
        break
