import torch
import torch.nn as nn
import torch.optim as optim
import torchvision 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import Discriminator, Generator, initialize_weights
from utils import gradient_penalty
import torch
from torch.utils.data import Dataset
import pickle
from transformers import BertTokenizer, BertModel
from transformers import BertModel, BertTokenizer,BertConfig
from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset
from gensim.models import KeyedVectors
import numpy as np
import clip
import torchvision.models as models
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import torch.optim as optim
import warnings
import fasttext
import fasttext.util
import torch.nn.functional as F
warnings.filterwarnings("ignore", category=FutureWarning)

class ResNet101Layer4(nn.Module):
    def __init__(self):
        super(ResNet101Layer4, self).__init__()
        resnet101 = models.resnet101(pretrained=True)
        
        # Keep only the layers up to layer 4 (excluding layer 4)
        self.layer4 = nn.Sequential(*list(resnet101.layer4.children()))

    def forward(self, x):
        x = self.layer4(x)
        return x    

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN101", device=device)
model.eval()
class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()

        # Create ResNet101Layer4
        self.resnet_layer4 = ResNet101Layer4()

        # Create ClipAttentionPooling
        self.attention_pooling = model.visual.attnpool

        # Additional layers if needed for your specific task

    def forward(self, x):
        # Forward through ResNet101Layer4
        # x_layer4 = self.resnet_layer4(x)
        x= torch.tensor(x,dtype=torch.float16)

        x_attention = self.attention_pooling(x)

        return x_attention
    

pretrained_weights_path = '/u/student/2022/cs22mtech14005/Thesis1/GAN/custom_model.pth'
pretrained_state_dict = torch.load(pretrained_weights_path)

model1 = CombinedModel().to('cuda')
model1.load_state_dict(pretrained_state_dict)
model1.eval()
#hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Learning_rate = 0.0001
Batch_size = 100
Image_size = 14
channels_img = 1024
z_dim = 196
Num_classes = 6
Gen_embedding = 196
Num_epochs = 100
critic_iterations = 4
lambda_gp = 10
Features_gen = 50
Features_disc = 100

classes = ['car', 'rider', 'truck', 'bike', 'person', 'motor']
class_mapping = {class_label: index for index, class_label in enumerate(classes)}
encoder = OneHotEncoder(sparse=False, categories='auto')
def custom_one_hot_encode(class_labels, class_mapping):
    one_hot_encodings = []
    for class_label in class_labels:
        class_index = class_mapping[class_label]
        one_hot_encoding = F.one_hot(torch.tensor(class_index), num_classes=len(class_mapping)).float()
        one_hot_encodings.append(one_hot_encoding)
    x = torch.stack(one_hot_encodings)
    x = x.to('cuda')
    return x

transforms = transforms.Compose(
    [
        transforms.Resize((Image_size,Image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(channels_img)],[0.5 for _ in range(channels_img)]),
    ]
)

model_path = 'cc.en.300.bin'
fasttext.util.download_model('en', if_exists='ignore')

# Load the downloaded model
ft = fasttext.load_model(model_path)

# Get the word vectors
words = ft.get_words()
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Retrieve the data and label at the specified index
        data_sample = self.data[index]
        label = self.labels[index]

        data_sample = torch.tensor(data_sample, dtype=torch.float32)

        # Get BERT word embeddings
        word_embeddings = semantics[words.index(label)]
        return data_sample, word_embeddings,self.labels[index]


print('here1')
with open('/u/student/2022/cs22mtech14005/Thesis1/ZS/domaingen/rois_file.pkl', 'rb') as file:
    rois = pickle.load(file)
with open('/u/student/2022/cs22mtech14005/Thesis1/ZS/domaingen/labels_file.pkl', 'rb') as file:
    labels = pickle.load(file)    
with open('/u/student/2022/cs22mtech14005/Thesis1/semantics.pkl', 'rb') as file:
    semantics = pickle.load(file)   
    
rois1 = []
labels1 = []
for i in range(rois.__len__()):
  if labels[i] != 'motor':
      rois1.append(rois[i])
      labels1.append(labels[i])
rois = rois1
labels = labels1
print('------------------------------------------',rois.__len__())

# Create a DataLoader for batching and shuffling
custom_dataset = CustomDataset(rois, labels)
data_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=Batch_size, shuffle=True)

gen = Generator(z_dim,channels_img,Features_gen,Image_size,Gen_embedding).to(device)
critic = Discriminator(channels_img,Features_disc,Image_size).to(device)

initialize_weights(gen)
initialize_weights(critic)

opt_gen = torch.optim.Adam(gen.parameters(), lr=Learning_rate, betas=(0.5, 0.999))
opt_disc = torch.optim.Adam(critic.parameters(), lr=Learning_rate, betas=(0.5, 0.999))

# fixed_noise = torch.randn(10,z_dim,1,1).to(device)
step = 0

gen.train()
critic.train()
print('hello')
for param in model.parameters():
    param.requires_grad = False 
for param in model1.parameters():
    param.requires_grad = False     
classes_reshaped = np.array(classes).reshape(-1, 1)
criterion = nn.CrossEntropyLoss()
# opt_bce = optim.Adam(gen.parameters(), lr=0.001,betas=(0.0,0.9))
gen_ep = []
disc_ep = []
class_ep = []
for epoch in range(Num_epochs):
    batch_idx = 1
    gen_loss_ep = 0
    disc_loss_ep = 0
    class_loss_ep = 0
    for real,labels,one_hot in data_loader:
        real = real.to(device)
        labels = labels.to(device)
        one_hot_encoded = custom_one_hot_encode(one_hot, class_mapping)
        #Disc training
        temp = 0
        for _ in range(critic_iterations):
            noise = torch.randn((real.shape[0],z_dim,1,1)).to(device)
            fake = gen(noise,labels)
            disc_labels = labels.view(real.shape[0],4,7,7).to(device)
            critic_real = critic(real,disc_labels).reshape(-1)
            critic_fake = critic(fake,disc_labels).reshape(-1)
            gp = gradient_penalty(critic,disc_labels,real,fake,device=device)
            loss_critic = (
                -(torch.mean(critic_real)-torch.mean(critic_fake))+lambda_gp*gp
                )
            opt_disc.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_disc.step()
            del fake
            del noise
            del disc_labels
            temp += loss_critic.item()
        
        disc_loss_ep += temp/5
        #gen training min -E[critic(gen_fake)]
        noise = torch.randn((real.shape[0],z_dim,1,1)).to(device)
        fake = gen(noise,labels).to(device)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)
        combined = model1(fake)
        with torch.no_grad():
                image_features = combined
                text_features = model.encode_text(text_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        image_features = torch.tensor(image_features,dtype=float)
        text_features = torch.tensor(text_features,dtype=float)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        similarity = torch.tensor(similarity, requires_grad=True, dtype=torch.float32).to('cuda')
        one_hot_encoded = one_hot_encoded.to('cuda')
        bce_loss = criterion(similarity,one_hot_encoded)
        disc_labels = labels.view(real.shape[0],4,7,7).to(device)
        output = critic(fake,disc_labels).reshape(-1)
        loss_gen = -torch.mean(output)
        final_loss = bce_loss + loss_gen
        opt_gen.zero_grad()
        final_loss.backward()
        opt_gen.step()
        gen_loss_ep += loss_gen.item()
        class_loss_ep += bce_loss.item()
        del similarity
        del combined
        del one_hot_encoded
        del image_features
        del text_features
        del fake
        del disc_labels
        del noise
    

        if batch_idx%5 == 0:
            print(
                f"Epoch [{epoch}/{Num_epochs-1}] Batch {batch_idx}/{len(data_loader)} \
                    Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )
            print(
                f"Discriminatory loss: {bce_loss}"
            )
        batch_idx += 1
    print(f"for epoch {epoch} \n disc loss {disc_loss_ep} \n gen loss  {gen_loss_ep} \n classification loss {class_loss_ep}")
    gen_ep.append(gen_loss_ep)
    disc_ep.append(disc_loss_ep)
    class_ep.append(class_loss_ep)

print('done')
with open('disc_loss.pkl', 'wb') as file:
    pickle.dump(disc_ep, file)
with open('gen_loss.pkl', 'wb') as file:
    pickle.dump(gen_ep, file)
with open('class_loss.pkl', 'wb') as file:
    pickle.dump(class_ep, file)
torch.save(critic.state_dict(), 'critic_model.pth')
torch.save(gen.state_dict(), 'generator_model.pth')