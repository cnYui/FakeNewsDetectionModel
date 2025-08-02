import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm.auto import tqdm
import os

# --- 使用Optuna找到的最佳超参数配置 ---
MODEL_PATH = '/root/models/model_cache/bert-base-chinese'
TRAIN_DATA_PATH = '/root/models/data/train.csv'
VAL_DATA_PATH = '/root/models/data/val.csv'
OUTPUT_DIR = '/root/models/testWordModel/bert_lstm_attention_best_model'
MAX_LEN = 128
BATCH_SIZE = 64
EPOCHS = 10
# 最佳超参数 (来自Optuna Trial 19)
LEARNING_RATE = 5.21e-05  # 优化后的学习率
LSTM_HIDDEN_DIM = 256     # 优化后的LSTM隐藏层维度
LSTM_LAYERS = 2           # 优化后的LSTM层数
DROPOUT = 0.33            # 优化后的Dropout比率
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EARLY_STOPPING_PATIENCE = 3

print(f"使用设备: {DEVICE}")
print(f"使用最佳超参数: LR={LEARNING_RATE}, LSTM_Dim={LSTM_HIDDEN_DIM}, LSTM_Layers={LSTM_LAYERS}, Dropout={DROPOUT}")

# --- Dataset Class ---
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len):
        self.data = pd.read_csv(file_path)
        if 'text' not in self.data.columns:
            raise ValueError(f"'text' column not found in {file_path}")
        self.data['text'] = self.data['text'].fillna('')
        self.texts = self.data['text'].tolist()
        self.labels = self.data['label'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = int(self.labels[item])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- Attention Mechanism ---
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention_dim = hidden_dim
        # Linear layer to project LSTM hidden states
        self.W = nn.Linear(hidden_dim, hidden_dim)
        # Linear layer to compute attention scores (context vector)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_dim)

        # Project hidden states
        # score shape: (batch_size, seq_len, hidden_dim)
        score = torch.tanh(self.W(lstm_output))

        # Compute attention scores
        # attention_score shape: (batch_size, seq_len, 1)
        attention_score = self.v(score)

        # Apply softmax to get weights
        # attention_weights shape: (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_score, dim=1)

        # Compute context vector (weighted sum of hidden states)
        # context_vector shape: (batch_size, seq_len, hidden_dim)
        context_vector = attention_weights * lstm_output

        # Sum across sequence dimension to get final context vector
        # context_vector shape: (batch_size, hidden_dim)
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector, attention_weights


# --- BERT+LSTM+Attention Model Definition ---
class BertLSTMClassifier(nn.Module):
    def __init__(self, bert_model_path, lstm_hidden_dim, num_labels, lstm_layers=1, dropout=0.1):
        super(BertLSTMClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_path)
        lstm_input_size = self.bert.config.hidden_size
        lstm_output_dim = lstm_hidden_dim * 2 # Bidirectional LSTM

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        # Add the attention layer
        self.attention = Attention(lstm_output_dim)
        self.dropout = nn.Dropout(dropout)
        # Classifier input is now the output of the attention layer
        self.classifier = nn.Linear(lstm_output_dim, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(sequence_output)
        # lstm_out shape: (batch_size, seq_len, lstm_hidden_dim * 2)

        # Attention forward pass
        # context_vector shape: (batch_size, lstm_hidden_dim * 2)
        context_vector, attention_weights = self.attention(lstm_out)

        # Apply dropout to the context vector
        dropped_output = self.dropout(context_vector)

        # Classification layer using the attention context vector
        logits = self.classifier(dropped_output)
        return logits

# --- Helper Functions ---
def compute_metrics(preds, labels):
    # Handle potential case where only one class is predicted
    try:
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    except ValueError:
        # If only one class present in labels or preds, binary metrics might fail
        precision, recall, f1 = 0.0, 0.0, 0.0
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# --- Main Training Logic ---
def train_model():
    print(f"加载tokenizer: {MODEL_PATH}")
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

    print(f"加载数据集...")
    train_dataset = TextDataset(TRAIN_DATA_PATH, tokenizer, MAX_LEN)
    val_dataset = TextDataset(VAL_DATA_PATH, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    num_labels = len(train_dataset.data['label'].unique())
    print(f"推断标签数量: {num_labels}")

    print(f"初始化BERT+LSTM+Attention模型...")
    model = BertLSTMClassifier(
        bert_model_path=MODEL_PATH,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        num_labels=num_labels,
        lstm_layers=LSTM_LAYERS,
        dropout=DROPOUT
    )
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    loss_fn = nn.CrossEntropyLoss().to(DEVICE)

    print("开始训练...")
    best_val_accuracy = 0.0
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=True, mininterval=5.0, maxinterval=10.0)

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            model.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)

            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'Training Loss': '{:.3f}'.format(loss.item())})

        avg_train_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch + 1} 平均训练损失: {avg_train_loss:.4f}")

        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0
        val_progress_bar = tqdm(val_loader, desc="验证中", leave=False)

        with torch.no_grad():
            for batch in val_progress_bar:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits, labels)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1).flatten()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_metrics = compute_metrics(val_preds, val_labels)

        print(f"验证损失: {avg_val_loss:.4f}")
        print(f"验证准确率: {val_metrics['accuracy']:.4f}")
        print(f"验证F1分数: {val_metrics['f1']:.4f}")

        if val_metrics['accuracy'] > best_val_accuracy:
            print(f"验证准确率提升 ({best_val_accuracy:.4f} --> {val_metrics['accuracy']:.4f}). 保存模型...")
            best_val_accuracy = val_metrics['accuracy']
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pt'))
            tokenizer.save_pretrained(OUTPUT_DIR)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"验证准确率未提升 {epochs_no_improve} 个epoch. 最佳准确率: {best_val_accuracy:.4f}")
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\n早停触发，在第 {epoch + 1} 个epoch后停止训练.")
                break

    print(f"训练完成. 最佳模型已保存到 {OUTPUT_DIR}，验证准确率: {best_val_accuracy:.4f}")

if __name__ == "__main__":
    train_model()
