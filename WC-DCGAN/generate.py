from model import Generator
import torch
import transformers
from transformers import BertTokenizer, BertModel
import torch
from custom import CombinedModel
import clip
import pickle
import fasttext
import fasttext.util
import numpy as np

Learning_rate = 1e-4
Batch_size = 5
Image_size = 14
channels_img = 1024
z_dim = 300
Num_classes = 6
Gen_embedding = 300
Num_epochs = 5
critic_iterations = 5
lambda_gp = 10
Features_gen = 50
Features_disc = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gen = Generator(z_dim,Gen_embedding).to(device)



bus_emb = []
model_path = '/u/student/2022/cs22mtech14005/Single-Source-domain-generalized-zero-shot-object-detection/WC-DCGAN/cc.en.300.bin'
fasttext.util.download_model('en', if_exists='ignore')

# Load the downloaded model
ft = fasttext.load_model(model_path)

# Get the word vectors
words = ft.get_words()


gen_weights_path = '/u/student/2022/cs22mtech14005/Single-Source-domain-generalized-zero-shot-object-detection/WC-DCGAN/generator_model_mid.pth'
gen_state_dict = torch.load(gen_weights_path)
gen.load_state_dict(gen_state_dict)
gen.eval()
word_vectors = np.array([ft.get_word_vector(word) for word in words])

for i in range(0,80000):
    noise = torch.randn((1,z_dim,1,1)).to(device)
    z = torch.tensor(word_vectors[words.index('bus')]).to(device)
    z = z.unsqueeze(0)
    vector = gen(noise,z)
    bus_emb.append(vector.squeeze(0).detach().cpu().numpy())
    print(f"Generated {i} samples")
 

with open('bus_emb.pkl', 'wb') as file:
    pickle.dump(bus_emb, file)
