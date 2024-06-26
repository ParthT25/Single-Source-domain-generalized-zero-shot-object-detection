import torch
import torch.nn as nn
import pickle
import numpy as np
import clip
import numpy as np
import warnings
import fasttext
import fasttext.util
import torch.nn.functional as F
from custom import CombinedModel,CustomDataset,custom_one_hot_encode
warnings.filterwarnings("ignore")

device = "cuda:6"
model, preprocess = clip.load("RN101", device=device)
model.transformer.requires_grad = False
model.visual.requires_grad = False
model.visual.attnpool.requires_grad = True


Learning_rate = 0.0001
Gen_embedding = 196
Num_epochs = 5
Batch_size = 160


classes = ['car','bus', 'rider', 'truck', 'bike', 'person', 'motor']
class_mapping = {class_label: index for index, class_label in enumerate(classes)}


pretrained_weights_path = '/u/student/2022/cs22mtech14005/Thesis1/zs2/domaingen/all_outs/diverse_weather/model_best.pth'
pretrained_state_dict = torch.load(pretrained_weights_path)
model.visual.attnpool.k_proj.bias.data = torch.tensor(pretrained_state_dict['model']['backbone.enc.attnpool.k_proj.bias'],dtype=torch.float16).to(device)
model.visual.attnpool.c_proj.bias.data = torch.tensor(pretrained_state_dict['model']['backbone.enc.attnpool.c_proj.bias'],dtype=torch.float16).to(device)
model.visual.attnpool.c_proj.weight.data = torch.tensor(pretrained_state_dict['model']['backbone.enc.attnpool.c_proj.weight'],dtype=torch.float16).to(device)
model.visual.attnpool.k_proj.weight.data = torch.tensor(pretrained_state_dict['model']['backbone.enc.attnpool.k_proj.weight'],dtype=torch.float16).to(device)
model.visual.attnpool.v_proj.bias.data = torch.tensor(pretrained_state_dict['model']['backbone.enc.attnpool.v_proj.bias'],dtype=torch.float16).to(device)
model.visual.attnpool.v_proj.weight.data = torch.tensor(pretrained_state_dict['model']['backbone.enc.attnpool.v_proj.weight'],dtype=torch.float16).to(device)
model.visual.attnpool.q_proj.bias.data = torch.tensor(pretrained_state_dict['model']['backbone.enc.attnpool.q_proj.bias'],dtype=torch.float16).to(device)
model.visual.attnpool.q_proj.weight.data = torch.tensor(pretrained_state_dict['model']['backbone.enc.attnpool.q_proj.weight'],dtype=torch.float16).to(device)
model.visual.attnpool.positional_embedding.data = torch.tensor(pretrained_state_dict['model']['backbone.enc.attnpool.positional_embedding'],dtype=torch.float16).to(device)
opt = torch.optim.SGD(model.visual.parameters(),lr=Learning_rate)
model.train()    

# print('Reading Data....')
with open('/u/student/2022/cs22mtech14005/Thesis1/zs2/domaingen/rois_file_final.pkl', 'rb') as file:
    rois = pickle.load(file)
with open('/u/student/2022/cs22mtech14005/Thesis1/zs2/domaingen/labels_file_final.pkl', 'rb') as file:
    labels = pickle.load(file)  
with open('bus_emb_of.pkl', 'rb') as file:
    bus_emb = pickle.load(file)

rois1 = []
labels1 = []
for i in range(rois.__len__()):
    if labels[i] != 'background':
        rois1.append(rois[i])
        labels1.append(labels[i])

rois = rois1
labels = labels1

model_path = '/u/student/2022/cs22mtech14005/Thesis1/GAN/cc.en.300.bin'
ft = fasttext.load_model(model_path)

# Get the word vectors
words = ft.get_words()

print(bus_emb[0].shape)
for i in range(0,int(bus_emb.__len__()/2)):
    rois.append(bus_emb[i])
    labels.append('bus')
del bus_emb


# word_vectors = np.array([ft.get_word_vector(word) for word in words])
        
        
custom_dataset = CustomDataset(rois, labels)
data_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=Batch_size, shuffle=True)
criterion = nn.CrossEntropyLoss()
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)
disc_ep = []
for epoch in range(Num_epochs):
    batch_idx = 1
    class_loss_ep = 0
    for real,labels,one_hot in data_loader:
        real = real.to(device)
        labels = labels.to(device)
        one_hot_encoded = custom_one_hot_encode(one_hot, class_mapping)
        real = torch.tensor(real,dtype=torch.float16)
        combined = model.visual.attnpool(real)
        
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
        combined = combined/combined.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * combined @ text_features.T).softmax(dim=-1)
        bce_loss = criterion(similarity, one_hot_encoded)
        opt.zero_grad()
        bce_loss.backward()
        opt.step() 
        class_loss_ep += bce_loss.item()
        if batch_idx%50 == 0:
            print(
                f"Epoch [{epoch}/{Num_epochs-1}] Batch {batch_idx}/{len(data_loader)} \
                    Discriminatory loss: {bce_loss}"
            )

        
        batch_idx += 1
    print(f"for epoch {epoch} classification loss {class_loss_ep}")
    disc_ep.append(class_loss_ep)
    

with open('retrain_loss.pkl', 'wb') as file:
    pickle.dump(disc_ep, file)

pretrained_state_dict['model']['roi_heads.clip_im_predictor.visual_enc.attnpool.k_proj.bias'] = model.visual.attnpool.k_proj.bias.data
pretrained_state_dict['model']['roi_heads.clip_im_predictor.visual_enc.attnpool.c_proj.bias'] = model.visual.attnpool.c_proj.bias.data
pretrained_state_dict['model']['roi_heads.clip_im_predictor.visual_enc.attnpool.c_proj.weight'] = model.visual.attnpool.c_proj.weight.data
pretrained_state_dict['model']['roi_heads.clip_im_predictor.visual_enc.attnpool.k_proj.weight'] = model.visual.attnpool.k_proj.weight.data
pretrained_state_dict['model']['roi_heads.clip_im_predictor.visual_enc.attnpool.v_proj.bias'] = model.visual.attnpool.v_proj.bias.data
pretrained_state_dict['model']['roi_heads.clip_im_predictor.visual_enc.attnpool.v_proj.weight'] = model.visual.attnpool.v_proj.weight.data
pretrained_state_dict['model']['roi_heads.clip_im_predictor.visual_enc.attnpool.q_proj.bias'] = model.visual.attnpool.q_proj.bias.data
pretrained_state_dict['model']['roi_heads.clip_im_predictor.visual_enc.attnpool.q_proj.weight'] = model.visual.attnpool.q_proj.weight.data
pretrained_state_dict['model']['roi_heads.clip_im_predictor.visual_enc.attnpool.positional_embedding'] = model.visual.attnpool.positional_embedding.data
print('--------------------------------------------------------------------------------')
print('Saving...')
torch.save(pretrained_state_dict, 'updated_clipattn_vani.pth')