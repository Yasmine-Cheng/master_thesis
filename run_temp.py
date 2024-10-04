# # import torch
# # from torch.utils.data import Dataset, DataLoader
# # from transformers import BertTokenizer
# # import torch.optim as optim
# # import torch.nn.functional as F
# # from torch.distributions.kl import kl_divergence
# # from torch.distributions import Normal
# # from temp_G import ThreadsGANGenerator
# # from temp_D import ThreadsGANDiscriminator, BertFeatureExtractor
# # import pandas as pd
# # import os

# # class SimplePTTDataset(Dataset):
# #     def __init__(self, dataframe):
# #         self.data = dataframe
# #     def __len__(self):
# #         return len(self.data)
# #     def __getitem__(self, idx):
# #         post = self.data.iloc[idx]['post']
# #         prev_resp = self.data.iloc[idx]['prev_resp']
# #         next_resp = self.data.iloc[idx]['next_resp']
# #         topic = self.data.iloc[idx]['topic']
# #         return {
# #             'post': post,
# #             'prev_resp': prev_resp,
# #             'next_resp': next_resp,
# #             'topic': topic}
# # def create_dataloader(dataframe, batch_size=32):
# #     dataset = SimplePTTDataset(dataframe)
# #     dataloader = DataLoader(dataset, batch_size=batch_size)
# #     return dataloader
# # def generator_loss(kl_real, kl_fake, reconstruction_term, alpha=1):
# #     kl_loss = -alpha * kl_divergence(kl_fake, kl_real)
# #     return kl_loss + reconstruction_term
# # def discriminator_loss(m_t, m_g):
# #     return torch.log(m_t) + torch.log(1 - m_g)
# # def train_gan(generator, discriminator, dataloader, num_epochs=50, lambda_=1.0, alpha=1, save_dir="./models", best_model_name="best_model_v2.pth", gen_steps=3):
# #     optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-6)
# #     optimizer_G = optim.Adam(generator.parameters(), lr=1e-15)
# #     generator.train()
# #     discriminator.train()
# #     best_loss_G = float('inf')
# #     if not os.path.exists(save_dir):
# #         os.makedirs(save_dir)
# #     for epoch in range(num_epochs):
# #         for batch in dataloader:
# #             post_text_ids = tokenizer(batch['post'][0], return_tensors='pt', truncation=True)['input_ids'].to(device)
# #             prev_text_ids = tokenizer(batch['prev_resp'][0], return_tensors='pt', truncation=True)['input_ids'].to(device)
# #             next_text_ids = tokenizer(batch['next_resp'][0], return_tensors='pt', truncation=True)['input_ids'].to(device)
# #             topic_ids = tokenizer(batch['topic'][0], return_tensors='pt', truncation=True)['input_ids'].to(device)
# #             for _ in range(gen_steps):
# #                 generated_sentence = generator(post_text_ids, prev_text_ids, next_text_ids, topic_ids)
# #                 H_next_real = feature_extractor.extract_hidden_state(batch['next_resp'][0]).to(device)
# #                 H_next_generated = feature_extractor.extract_hidden_state(generated_sentence).to(device)
# #                 H_prev = feature_extractor.extract_hidden_state(batch['prev_resp'][0]).to(device)
# #                 H_next_real.requires_grad_()
# #                 H_next_generated.requires_grad_()
# #                 H_prev.requires_grad_()
# #                 kl_real = Normal(H_next_real.mean(), H_next_real.std())
# #                 kl_fake = Normal(H_next_generated.mean(), H_next_generated.std())
# #                 reconstruction_term = F.mse_loss(H_next_generated, H_next_real)
# #                 L_G = generator_loss(kl_real, kl_fake, reconstruction_term, alpha=alpha)
# #                 optimizer_G.zero_grad()
# #                 L_G.backward()
# #                 torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
# #                 optimizer_G.step()
# #             m_g = discriminator(H_next_generated, H_next_real, H_prev).to(device)
# #             m_t = discriminator(H_next_real, H_next_real, H_prev).to(device)
# #             L_D = discriminator_loss(m_t, m_g)
# #             optimizer_D.zero_grad()
# #             L_D.backward()
# #             torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
# #             optimizer_D.step()
# #         print(f"Epoch [{epoch+1}/{num_epochs}], Loss_G: {L_G.item():.4f}, Loss_D: {L_D.item():.4f}")
# #         if L_G.item() < best_loss_G:
# #             best_loss_G = L_G.item()
# #             torch.save({
# #                 'epoch': epoch + 1,
# #                 'generator_state_dict': generator.state_dict(),
# #                 'discriminator_state_dict': discriminator.state_dict(),
# #                 'optimizer_G_state_dict': optimizer_G.state_dict(),
# #                 'optimizer_D_state_dict': optimizer_D.state_dict(),
# #                 'loss_G': best_loss_G,
# #             }, os.path.join(save_dir, best_model_name))
# #             print(f"New best model saved at epoch {epoch+1} with Loss_G: {best_loss_G:.4f}")


# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # data = pd.read_csv('../dataset/ptt_train_preprocessed.csv')
# # tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# # batch_size = 500
# # dataloader = create_dataloader(data, batch_size=batch_size)
# # generator = ThreadsGANGenerator().to(device)
# # discriminator = ThreadsGANDiscriminator().to(device)
# # feature_extractor = BertFeatureExtractor()
# # train_gan(generator, discriminator, dataloader, num_epochs=10)

# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import BertTokenizer
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.distributions.kl import kl_divergence
# from torch.distributions import Normal
# from temp_G import ThreadsGANGenerator
# from temp_D import ThreadsGANDiscriminator, BertFeatureExtractor
# import pandas as pd
# import os

# class SimplePTTDataset(Dataset):
#     def __init__(self, dataframe):
#         self.data = dataframe
#     def __len__(self):
#         return len(self.data)
#     def __getitem__(self, idx):
#         post = self.data.iloc[idx]['post']
#         prev_resp = self.data.iloc[idx]['prev_resp']
#         next_resp = self.data.iloc[idx]['next_resp']
#         topic = self.data.iloc[idx]['topic']
#         return {
#             'post': post,
#             'prev_resp': prev_resp,
#             'next_resp': next_resp,
#             'topic': topic}

# def create_dataloader(dataframe, batch_size=32):
#     dataset = SimplePTTDataset(dataframe)
#     dataloader = DataLoader(dataset, batch_size=batch_size)
#     return dataloader

# def generator_loss(kl_real, kl_fake, reconstruction_term, alpha=1):
#     kl_loss = -alpha * kl_divergence(kl_fake, kl_real)
#     return kl_loss + reconstruction_term

# def discriminator_loss(m_t, m_g):
#     return torch.log(m_t) + torch.log(1 - m_g)

# def train_gan(generator, discriminator, dataloader, num_epochs=50, lambda_=1.0, alpha=1, save_dir="./models", best_model_name="best_model_v2.pth", gen_steps=3):
#     optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-6)
#     optimizer_G = optim.Adam(generator.parameters(), lr=1e-15)
    
#     # 使用 DataParallel
#     generator = torch.nn.DataParallel(generator)
#     discriminator = torch.nn.DataParallel(discriminator)
    
#     generator.train()
#     discriminator.train()
    
#     best_loss_G = float('inf')
    
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
    
#     for epoch in range(num_epochs):
#         for batch in dataloader:
#             post_text_ids = tokenizer(batch['post'][0], return_tensors='pt', truncation=True)['input_ids'].to(device)
#             prev_text_ids = tokenizer(batch['prev_resp'][0], return_tensors='pt', truncation=True)['input_ids'].to(device)
#             next_text_ids = tokenizer(batch['next_resp'][0], return_tensors='pt', truncation=True)['input_ids'].to(device)
#             topic_ids = tokenizer(batch['topic'][0], return_tensors='pt', truncation=True)['input_ids'].to(device)
            
#             for _ in range(gen_steps):
#                 generated_sentence = generator(post_text_ids, prev_text_ids, next_text_ids, topic_ids)
#                 H_next_real = feature_extractor.extract_hidden_state(batch['next_resp'][0]).to(device)
#                 H_next_generated = feature_extractor.extract_hidden_state(generated_sentence).to(device)
#                 H_prev = feature_extractor.extract_hidden_state(batch['prev_resp'][0]).to(device)
                
#                 H_next_real.requires_grad_()
#                 H_next_generated.requires_grad_()
#                 H_prev.requires_grad_()
                
#                 kl_real = Normal(H_next_real.mean(), H_next_real.std())
#                 kl_fake = Normal(H_next_generated.mean(), H_next_generated.std())
#                 reconstruction_term = F.mse_loss(H_next_generated, H_next_real)
                
#                 L_G = generator_loss(kl_real, kl_fake, reconstruction_term, alpha=alpha)
                
#                 optimizer_G.zero_grad()
#                 L_G.backward()
#                 torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
#                 optimizer_G.step()
            
#             m_g = discriminator(H_next_generated, H_next_real, H_prev).to(device)
#             m_t = discriminator(H_next_real, H_next_real, H_prev).to(device)
            
#             L_D = discriminator_loss(m_t, m_g)
#             optimizer_D.zero_grad()
#             L_D.backward()
#             torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
#             optimizer_D.step()
        
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss_G: {L_G.item():.4f}, Loss_D: {L_D.item():.4f}")
        
#         if L_G.item() < best_loss_G:
#             best_loss_G = L_G.item()
#             torch.save({
#                 'epoch': epoch + 1,
#                 'generator_state_dict': generator.module.state_dict(),  # 取出 DataParallel 中的模型
#                 'discriminator_state_dict': discriminator.module.state_dict(),
#                 'optimizer_G_state_dict': optimizer_G.state_dict(),
#                 'optimizer_D_state_dict': optimizer_D.state_dict(),
#                 'loss_G': best_loss_G,
#             }, os.path.join(save_dir, best_model_name))
#             print(f"New best model saved at epoch {epoch+1} with Loss_G: {best_loss_G:.4f}")


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# data = pd.read_csv('../dataset/ptt_train_preprocessed.csv')
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# batch_size = 500
# dataloader = create_dataloader(data, batch_size=batch_size)
# generator = ThreadsGANGenerator().to(device)
# discriminator = ThreadsGANDiscriminator().to(device)
# feature_extractor = BertFeatureExtractor()

# train_gan(generator, discriminator, dataloader, num_epochs=10)


# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import BertTokenizer
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.distributions.kl import kl_divergence
# from torch.distributions import Normal
# from temp_G import ThreadsGANGenerator
# from temp_D import ThreadsGANDiscriminator, BertFeatureExtractor
# import pandas as pd
# import os

# class SimplePTTDataset(Dataset):
#     def __init__(self, dataframe):
#         self.data = dataframe
#     def __len__(self):
#         return len(self.data)
#     def __getitem__(self, idx):
#         post = self.data.iloc[idx]['post']
#         prev_resp = self.data.iloc[idx]['prev_resp']
#         next_resp = self.data.iloc[idx]['next_resp']
#         topic = self.data.iloc[idx]['topic']
#         return {
#             'post': post,
#             'prev_resp': prev_resp,
#             'next_resp': next_resp,
#             'topic': topic}
# def create_dataloader(dataframe, batch_size=32):
#     dataset = SimplePTTDataset(dataframe)
#     dataloader = DataLoader(dataset, batch_size=batch_size)
#     return dataloader
# def generator_loss(kl_real, kl_fake, reconstruction_term, alpha=1):
#     kl_loss = -alpha * kl_divergence(kl_fake, kl_real)
#     return kl_loss + reconstruction_term
# def discriminator_loss(m_t, m_g):
#     return torch.log(m_t) + torch.log(1 - m_g)
# def train_gan(generator, discriminator, dataloader, num_epochs=50, lambda_=1.0, alpha=1, save_dir="./models", best_model_name="best_model_v2.pth", gen_steps=3):
#     optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-6)
#     optimizer_G = optim.Adam(generator.parameters(), lr=1e-15)
#     generator.train()
#     discriminator.train()
#     best_loss_G = float('inf')
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     for epoch in range(num_epochs):
#         for batch in dataloader:
#             post_text_ids = tokenizer(batch['post'][0], return_tensors='pt', truncation=True)['input_ids'].to(device)
#             prev_text_ids = tokenizer(batch['prev_resp'][0], return_tensors='pt', truncation=True)['input_ids'].to(device)
#             next_text_ids = tokenizer(batch['next_resp'][0], return_tensors='pt', truncation=True)['input_ids'].to(device)
#             topic_ids = tokenizer(batch['topic'][0], return_tensors='pt', truncation=True)['input_ids'].to(device)
#             for _ in range(gen_steps):
#                 generated_sentence = generator(post_text_ids, prev_text_ids, next_text_ids, topic_ids)
#                 H_next_real = feature_extractor.extract_hidden_state(batch['next_resp'][0]).to(device)
#                 H_next_generated = feature_extractor.extract_hidden_state(generated_sentence).to(device)
#                 H_prev = feature_extractor.extract_hidden_state(batch['prev_resp'][0]).to(device)
#                 H_next_real.requires_grad_()
#                 H_next_generated.requires_grad_()
#                 H_prev.requires_grad_()
#                 kl_real = Normal(H_next_real.mean(), H_next_real.std())
#                 kl_fake = Normal(H_next_generated.mean(), H_next_generated.std())
#                 reconstruction_term = F.mse_loss(H_next_generated, H_next_real)
#                 L_G = generator_loss(kl_real, kl_fake, reconstruction_term, alpha=alpha)
#                 optimizer_G.zero_grad()
#                 L_G.backward()
#                 torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
#                 optimizer_G.step()
#             m_g = discriminator(H_next_generated, H_next_real, H_prev).to(device)
#             m_t = discriminator(H_next_real, H_next_real, H_prev).to(device)
#             L_D = discriminator_loss(m_t, m_g)
#             optimizer_D.zero_grad()
#             L_D.backward()
#             torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
#             optimizer_D.step()
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss_G: {L_G.item():.4f}, Loss_D: {L_D.item():.4f}")
#         if L_G.item() < best_loss_G:
#             best_loss_G = L_G.item()
#             torch.save({
#                 'epoch': epoch + 1,
#                 'generator_state_dict': generator.state_dict(),
#                 'discriminator_state_dict': discriminator.state_dict(),
#                 'optimizer_G_state_dict': optimizer_G.state_dict(),
#                 'optimizer_D_state_dict': optimizer_D.state_dict(),
#                 'loss_G': best_loss_G,
#             }, os.path.join(save_dir, best_model_name))
#             print(f"New best model saved at epoch {epoch+1} with Loss_G: {best_loss_G:.4f}")


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# data = pd.read_csv('../dataset/ptt_train_preprocessed.csv')
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# batch_size = 500
# dataloader = create_dataloader(data, batch_size=batch_size)
# generator = ThreadsGANGenerator().to(device)
# discriminator = ThreadsGANDiscriminator().to(device)
# feature_extractor = BertFeatureExtractor()
# train_gan(generator, discriminator, dataloader, num_epochs=10)

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal
from temp_G import ThreadsGANGenerator
from temp_D import ThreadsGANDiscriminator, BertFeatureExtractor
import pandas as pd
import os

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
            'topic': topic
        }

def create_dataloader(dataframe, batch_size=32):
    dataset = SimplePTTDataset(dataframe)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

def generator_loss(fake_score):
    # Generator tries to minimize discriminator's prediction for fake samples
    return -torch.mean(fake_score)

def discriminator_loss(real_score, fake_score):
    # Discriminator loss based on Wasserstein distance
    return torch.mean(fake_score) - torch.mean(real_score)

# Apply weight clipping after discriminator update to enforce Lipschitz constraint
def clip_discriminator_weights(discriminator, clip_value=0.01):
    for p in discriminator.parameters():
        p.data.clamp_(-clip_value, clip_value)

def train_wgan(generator, discriminator, dataloader, num_epochs=30, clip_value=0.1, save_dir="./models", best_model_name="best_model_v12.pth", gen_steps=3):
    optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-17)
    optimizer_G = optim.Adam(generator.parameters(), lr=1e-12)
    
    # generator = torch.nn.DataParallel(generator)
    # discriminator = torch.nn.DataParallel(discriminator)
    
    generator.train()
    discriminator.train()
    
    best_loss_G = float('inf')
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            post_text_ids = tokenizer(batch['post'][0], return_tensors='pt', truncation=True)['input_ids'].to(device)
            prev_text_ids = tokenizer(batch['prev_resp'][0], return_tensors='pt', truncation=True)['input_ids'].to(device)
            next_text_ids = tokenizer(batch['next_resp'][0], return_tensors='pt', truncation=True)['input_ids'].to(device)
            topic_ids = tokenizer(batch['topic'][0], return_tensors='pt', truncation=True)['input_ids'].to(device)

            # ---------------------
            # Train Discriminator
            # ---------------------
            for _ in range(gen_steps):
                generated_sentence = generator(post_text_ids, prev_text_ids, next_text_ids, topic_ids)
                H_next_real = feature_extractor.extract_hidden_state(batch['next_resp'][0]).to(device)
                H_next_generated = feature_extractor.extract_hidden_state(generated_sentence).to(device)
                H_prev = feature_extractor.extract_hidden_state(batch['prev_resp'][0]).to(device)

                H_next_real.requires_grad_()
                H_next_generated.requires_grad_()
                H_prev.requires_grad_()

                # Real and fake scores from the discriminator
                real_score = discriminator(H_next_real, H_next_real, H_prev)
                fake_score = discriminator(H_next_generated, H_next_real, H_prev)

                # Discriminator loss (no gradient penalty, only Wasserstein loss)
                L_D = discriminator_loss(real_score, fake_score)

                # Backpropagate and optimize discriminator
                optimizer_D.zero_grad()
                L_D.backward()
                # torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                optimizer_D.step()


                # for param in discriminator.parameters():
                #     print("param=%s, grad=%s" % (param.data.item(), param.grad.item()))


                # Apply weight clipping to enforce Lipschitz constraint
                clip_discriminator_weights(discriminator, clip_value=clip_value)

            # ---------------------
            # Train Generator
            # ---------------------
            generated_sentence = generator(post_text_ids, prev_text_ids, next_text_ids, topic_ids)
            H_next_generated = feature_extractor.extract_hidden_state(generated_sentence).to(device)
            H_prev = feature_extractor.extract_hidden_state(batch['prev_resp'][0]).to(device)

            fake_score = discriminator(H_next_generated, H_next_generated, H_prev)

            # Generator loss
            L_G = generator_loss(fake_score)

            # Backpropagate and optimize generator
            optimizer_G.zero_grad()
            L_G.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer_G.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss_G: {L_G.item():.4f}, Loss_D: {L_D.item():.4f}")

        if L_G.item() < best_loss_G:
            best_loss_G = L_G.item()
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'loss_G': best_loss_G,
            }, os.path.join(save_dir, best_model_name))
            print(f"New best model saved at epoch {epoch+1} with Loss_G: {best_loss_G:.4f}")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = pd.read_csv('../dataset/ptt_train_preprocessed.csv')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
batch_size = 5000
dataloader = create_dataloader(data, batch_size=batch_size)
generator = ThreadsGANGenerator().to(device)
discriminator = ThreadsGANDiscriminator().to(device)
feature_extractor = BertFeatureExtractor()

train_wgan(generator, discriminator, dataloader, num_epochs=10)