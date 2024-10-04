import torch
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from torch.utils.data import Dataset, DataLoader
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from bert_score import score as bertscore
from temp_G import ThreadsGANGenerator
from temp_D import ThreadsGANDiscriminator, BertFeatureExtractor
from transformers import BertTokenizer
import os

# 讀取資料集的 Dataset 類
class SimplePTTDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        post = self.data.iloc[idx]['post']
        prev_resp = self.data.iloc[idx]['prev_resp']
        next_resp = self.data.iloc[idx]['next_resp']
        topic = self.data.iloc[idx]['topic']
        return {
            'post': post,
            'prev_resp': prev_resp,
            'next_resp': next_resp,
            'topic': topic}

# 創建 DataLoader
def create_dataloader(dataframe, batch_size=16):
    dataset = SimplePTTDataset(dataframe)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

def load_model_and_evaluate(generator, discriminator, test_dataloader, save_dir="./models", best_model_name="best_model.pth"):
    # 加載保存的最佳模型
    checkpoint = torch.load(os.path.join(save_dir, best_model_name))
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    # 設置模型為評估模式
    generator.eval()
    discriminator.eval()

    # 設置 ROUGE 計算器
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    bleu_scores = []
    rouge_scores = {'rouge1': [], 'rougeL': []}
    reference_sentences = []
    generated_sentences = []

    # 遍歷測試資料集
    for batch in test_dataloader:
        with torch.no_grad():
            # 將資料轉移到 GPU
            post_text_ids = tokenizer(batch['post'][0], return_tensors='pt', truncation=True)['input_ids'].to(device)
            prev_text_ids = tokenizer(batch['prev_resp'][0], return_tensors='pt', truncation=True)['input_ids'].to(device)
            next_text_ids = tokenizer(batch['next_resp'][0], return_tensors='pt', truncation=True)['input_ids'].to(device)
            topic_ids = tokenizer(batch['topic'][0], return_tensors='pt', truncation=True)['input_ids'].to(device)

            # 生成器生成句子，保持句子為字串類型
            generated_sentence = generator(post_text_ids, prev_text_ids, next_text_ids, topic_ids, max_length=50)  # 保持 generated_sentence 為字符串
            print(generated_sentence)
            # 真實的下一個回應
            reference_sentence = batch['next_resp'][0]
            print(reference_sentence)
            # 計算 BLEU 分數
            reference_tokens = [reference_sentence.split()]  # 參考的回應拆分為詞
            generated_tokens = generated_sentence.split()  # 生成的句子拆分為詞
            bleu_score = sentence_bleu(reference_tokens, generated_tokens)
            bleu_scores.append(bleu_score)

            # 計算 ROUGE 分數
            rouge_score = scorer.score(reference_sentence, generated_sentence)
            rouge_scores['rouge1'].append(rouge_score['rouge1'].fmeasure)
            rouge_scores['rougeL'].append(rouge_score['rougeL'].fmeasure)

            # 保存句子以便之後計算 BERTScore
            reference_sentences.append(reference_sentence)
            generated_sentences.append(generated_sentence)

    # 計算 BLEU 和 ROUGE 的平均分數
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_rouge1 = sum(rouge_scores['rouge1']) / len(rouge_scores['rouge1'])
    avg_rougeL = sum(rouge_scores['rougeL']) / len(rouge_scores['rougeL'])

    # 計算 BERTScore
    P, R, F1 = bertscore(generated_sentences, reference_sentences, lang='zh', device=device)
    avg_bertscore_f1 = F1.mean().item()

    print(f"Average BLEU Score: {avg_bleu:.4f}")
    print(f"Average ROUGE-1 F-Score: {avg_rouge1:.4f}")
    print(f"Average ROUGE-L F-Score: {avg_rougeL:.4f}")
    print(f"Average BERTScore F1: {avg_bertscore_f1:.4f}")

# 設定使用 GPU 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 讀取測試資料集
test_data = pd.read_csv('../dataset/testdata/ptt_test_preprocessed.csv')
test_data = test_data.head(200)

# 創建測試資料集的 DataLoader
test_dataloader = create_dataloader(test_data, batch_size=32)

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# 將生成器和判別器移動到 GPU
generator = ThreadsGANGenerator().to(device)
discriminator = ThreadsGANDiscriminator().to(device)

# 請注意，因為 BertFeatureExtractor 不是 PyTorch 模型，所以不需要使用 .to(device)
feature_extractor = BertFeatureExtractor()

# 使用保存的模型進行評估
load_model_and_evaluate(generator, discriminator, test_dataloader, save_dir="./models", best_model_name="best_model_v11.pth")
