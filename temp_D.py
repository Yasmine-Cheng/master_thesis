import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class ThreadsGANDiscriminator(nn.Module):
    def __init__(self, hidden_size=768, cnn_channels=128, kernel_size=3, num_features=5):
        super(ThreadsGANDiscriminator, self).__init__()
        self.cnn = nn.Conv1d(in_channels=hidden_size, out_channels=cnn_channels, kernel_size=kernel_size, padding=1)
        self.max_pool = nn.MaxPool1d(kernel_size=2)
        self.bn = torch.nn.BatchNorm1d(num_features)
        self.mlp_phi = nn.Linear(hidden_size, hidden_size)
        self.mlp_sigmoid = nn.Sequential(
            nn.Linear(cnn_channels, 1),
            nn.Sigmoid())
    def forward(self, H_next_generated, H_next_real, H_prev):
        H_phi_next_generated = self.mlp_phi(H_next_generated)
        H_phi_next_combined = torch.stack([H_phi_next_generated, H_prev], dim=2)
        H_cnn = self.cnn(H_phi_next_combined)
        H_cnn_pooled = self.max_pool(H_cnn).squeeze(2)
        m_g = self.mlp_sigmoid(H_cnn_pooled)
        return m_g
class BertFeatureExtractor:
    def __init__(self, bert_model_name='bert-base-multilingual-cased'):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name)
    def extract_hidden_state(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        hidden_state = outputs.last_hidden_state[:, 0, :]
        return hidden_state


if __name__ == "__main__":
    feature_extractor = BertFeatureExtractor()
    real_sentence = "中國就沒有要打  簽了幹嘛XD講兩句就叫人簽的 有沒有當過兵都不知道"
    generated_sentence = "中國是壞人，當然要揍他們啊"
    previous_sentence = "這種偷偷帶回去就可以啦!XDD MISAMO都能三天兩夜日本 MINA進入日本也無人發現 直到出現在發佈會上 OK der"
    H_next_real = feature_extractor.extract_hidden_state(real_sentence)
    H_next_generated = feature_extractor.extract_hidden_state(generated_sentence)
    H_prev = feature_extractor.extract_hidden_state(previous_sentence)
    discriminator = ThreadsGANDiscriminator()
    similarity_score = discriminator(H_next_generated, H_next_real, H_prev)
    print("Similarity score between generated and real sentences:", similarity_score.item())