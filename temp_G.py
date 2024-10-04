import torch
import json
from transformers import BertTokenizer, BertModel

def nucleus_sampling(logits, top_p=0.9, temperature=0.5, valid_token_ids=None):
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    logits[sorted_indices[sorted_indices_to_remove]] = -float('Inf')
    if valid_token_ids is not None:
        valid_token_set = set(valid_token_ids)
        for idx in sorted_indices:
            if idx.item() not in valid_token_set:
                logits[idx] = -float('Inf')
    probs = torch.softmax(logits, dim=-1)
    next_token_id = torch.multinomial(probs, 1).item()
    if valid_token_ids is not None and next_token_id not in valid_token_ids:
        raise ValueError(f"選擇了無效的 token: {next_token_id}")
    return next_token_id
class ThreadsGANGenerator(torch.nn.Module):
    def __init__(self, bert_model_name='bert-base-multilingual-cased', hidden_size=768, num_heads=8, num_layers=6, ff_dim=2048, max_len=20, num_features=5):
        super(ThreadsGANGenerator, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.mlp_prior = torch.nn.Linear(hidden_size, hidden_size)
        self.mlp_reco = torch.nn.Linear(hidden_size, hidden_size)
        self.mlp_topic = torch.nn.Linear(hidden_size, hidden_size)
        self.decoder = torch.nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=ff_dim)
        self.fc_out = torch.nn.Linear(hidden_size, hidden_size)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.vocab_size = self.tokenizer.vocab_size
        self.fc_final = torch.nn.Linear(hidden_size, self.vocab_size)
        self.max_len = max_len
        self.bn = torch.nn.BatchNorm1d(num_features)
        with open('vocab_filtered.json', 'r', encoding='utf-8') as f:
            self.filtered_vocab = json.load(f)
        self.valid_token_ids = list(self.filtered_vocab.values())
    def extract_bert_features(self, input_ids):
        outputs = self.bert(input_ids)
        cls_hidden_state = outputs.last_hidden_state[:, 0, :]
        return cls_hidden_state
    def forward(self, post_text_ids, prev_text_ids, next_text_ids, topic_ids, max_length=None, temperature=1.0, top_p=0.9):
        h_post = self.extract_bert_features(post_text_ids)
        h_prev = self.extract_bert_features(prev_text_ids)
        h_next = self.extract_bert_features(next_text_ids)
        z_prev = self.mlp_prior(h_prev)
        z_next = self.mlp_reco(h_next)
        h_topic = self.mlp_topic(h_post)
        combined_input = h_post + h_prev + z_prev + z_next + h_topic
        memory_input = combined_input.unsqueeze(0)
        if max_length is None:
            max_length = self.max_len
        generated_token_ids = []
        start_token_id = self.tokenizer.cls_token_id
        input_token_ids = torch.tensor([[start_token_id]], dtype=torch.long).to(post_text_ids.device)
        for _ in range(max_length):
            decoder_output = self.decoder(h_prev.unsqueeze(0), memory_input)
            output = self.fc_out(decoder_output)
            output = torch.softmax(output, dim=-1)
            output = self.fc_final(output.squeeze(0))
            next_token_id = nucleus_sampling(output[-1, :], top_p=top_p, temperature=temperature, valid_token_ids=self.valid_token_ids)
            generated_token_ids.append(next_token_id)
            if next_token_id == self.tokenizer.sep_token_id:
                break
            input_token_ids = torch.cat([input_token_ids, torch.tensor([[next_token_id]], dtype=torch.long).to(post_text_ids.device)], dim=1)
        generated_sentence = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        return generated_sentence

if __name__ == "__main__":
    generator = ThreadsGANGenerator(max_len=50)
    
    # 測試文本
    post_text = "文免役，蔣萬安椎間盤突出，只當幾天的補充兵）兩個人政治生涯都是從靠爸開始兩個人都大富大貴人家，卻都裝平民老百姓兩個人的爸爸都想靠兩岸議題牟利（連戰以前飛去中國好幾次了）付3千萬疏通陸官員\u3000台商控蔣孝嚴收錢不辦事2019/05/01 鏡週刊（雖然是鏡週刊，但已經被法院認證這些內容所以蔣孝嚴真的問題很大）吳姓台商1年半前因投資糾紛遭中國大陸境管，被要求繳交人民幣4千萬元（約新台幣1.8億元）才能回台，為此，吳找上由國民黨榮譽副主席蔣孝嚴擔任理事長的中國台商發展促進協會幫忙，先給財務長鳳姓女子人民幣700萬元（約新台幣3150萬元）作業費，並與蔣在大陸見面，但最後還是靠自己付錢給大陸官方才順利回台。不滿被控收3千萬不辦事 蔣孝嚴求償千萬敗訴2020/05/28 中時判決書都公告在網路上：綜上所述，原告（蔣孝嚴）主張被告（台商吳兆豐）所為如附表一編號1至3所示言論，以及使鏡週刊如附表二編號1至6之報導內容，依民法第184條前段規定，賠償原告1000萬元，及在蘋果日報、自由時報、中國時報、聯合報全國版之第1版報頭下刊登及鏡週刊之前3頁中以半頁之篇幅刊登如附表3之道歉啟事，為無理由，應予駁回。投機份子？連蔣兩家要自介逆？"
    prev_text = "[START]"
    next_text = "EE的爸爸不說一下嗎"
    topic_text = "[PAD]"
    
    # 編碼測試文本
    post_text_ids = generator.tokenizer(post_text, return_tensors='pt')['input_ids']
    prev_text_ids = generator.tokenizer(prev_text, return_tensors='pt')['input_ids']
    next_text_ids = generator.tokenizer(next_text, return_tensors='pt')['input_ids']
    topic_ids = generator.tokenizer(topic_text, return_tensors='pt')['input_ids']
    
    # 生成句子
    generated_sentence = generator(post_text_ids, prev_text_ids, next_text_ids, topic_ids, max_length=20)
    print("Generated Sentence:", generated_sentence)