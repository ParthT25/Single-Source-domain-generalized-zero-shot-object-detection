import torch
import torch.nn as nn
from model import Discriminator, Generator, initialize_weights
from utils import gradient_penalty
from custom import CombinedModel,CustomDataset,custom_one_hot_encode
import pickle
import numpy as np
import clip 
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import warnings
import fasttext
import random
warnings.filterwarnings("ignore")

device = "cuda:7" if torch.cuda.is_available() else "cpu"
Learning_rate     = 0.0001
Batch_size        = 100
channels_img      = 1024
z_dim             = 300
Gen_embedding     = 300
Num_epochs        = 32
critic_iterations = 10
lambda_gp         = 10
lambda_ce         = 1.0
lambda_mode       = 0.1
lambda_bus        = 0.25
thres_ep          = 10
bus_thresh        = 250
file_name         = 'output1.txt'
check = False
    

model, preprocess = clip.load("RN101", device=device)
model.eval()

model1 = CombinedModel().to(device)
pretrained_weights_path = 'domaingen/all_outs/diverse_weather/model_best.pth'
pretrained_state_dict = torch.load(pretrained_weights_path)
model1.attention_pooling.k_proj.bias.data = torch.tensor(pretrained_state_dict['model']['backbone.enc.attnpool.k_proj.bias'],dtype=torch.float16).to(device)
model1.attention_pooling.c_proj.bias.data = torch.tensor(pretrained_state_dict['model']['backbone.enc.attnpool.c_proj.bias'],dtype=torch.float16).to(device)
model1.attention_pooling.c_proj.weight.data = torch.tensor(pretrained_state_dict['model']['backbone.enc.attnpool.c_proj.weight'],dtype=torch.float16).to(device)
model1.attention_pooling.k_proj.weight.data = torch.tensor(pretrained_state_dict['model']['backbone.enc.attnpool.k_proj.weight'],dtype=torch.float16).to(device)
model1.attention_pooling.v_proj.bias.data = torch.tensor(pretrained_state_dict['model']['backbone.enc.attnpool.v_proj.bias'],dtype=torch.float16).to(device)
model1.attention_pooling.v_proj.weight.data = torch.tensor(pretrained_state_dict['model']['backbone.enc.attnpool.v_proj.weight'],dtype=torch.float16).to(device)
model1.attention_pooling.q_proj.bias.data = torch.tensor(pretrained_state_dict['model']['backbone.enc.attnpool.q_proj.bias'],dtype=torch.float16).to(device)
model1.attention_pooling.q_proj.weight.data = torch.tensor(pretrained_state_dict['model']['backbone.enc.attnpool.q_proj.weight'],dtype=torch.float16).to(device)
model1.attention_pooling.positional_embedding.data = torch.tensor(pretrained_state_dict['model']['backbone.enc.attnpool.positional_embedding'],dtype=torch.float16).to(device)
model1.eval()



classes = ['bus','car', 'rider', 'truck', 'bike', 'person', 'motor','background']
classes1 = ['bus','car', 'rider', 'truck', 'bike', 'person', 'motor','background']
missing = 'bus'
class_mapping = {class_label: index for index, class_label in enumerate(classes)}
encoder = OneHotEncoder(sparse=False, categories='auto')


model_path = 'WC-DCGAN/cc.en.300.bin'
ft = fasttext.load_model(model_path)
words = ft.get_words()


with open(file_name, 'w') as op_file:
    print('Reading ROIs Data',file = op_file,flush=True)
    print('Reading ROIs Data')
    with open('domaingen/rois_file_final.pkl', 'rb') as file:
        rois = pickle.load(file)
    with open('domaingen/labels_file_final.pkl', 'rb') as file:
        labels = pickle.load(file) 
    print('--------------------------------------------------------')
    print(f'Training WC-DCGAN on {rois.__len__()} instances of ROIs')
    print(f'Training WC-DCGAN on {rois.__len__()} instances of ROIs',file=op_file,flush=True)
    print('--------------------------------------------------------')

    custom_dataset = CustomDataset(rois, labels)
    data_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=Batch_size, shuffle=True)

    gen = Generator(z_dim,Gen_embedding).to(device)
    critic = Discriminator().to(device)
    print(gen,file=op_file,flush=True)

    initialize_weights(gen)
    initialize_weights(critic)


    opt_gen = torch.optim.Adam(gen.parameters(), lr=Learning_rate,betas=(0.5,0.999))
    opt_disc = torch.optim.Adam(critic.parameters(), lr=Learning_rate,betas=(0.5,0.999))

    word_vectors = np.array([ft.get_word_vector(word) for word in words])
    gen.train()
    critic.train()
    model.eval()
    model1.eval()

    criterion = nn.CrossEntropyLoss()

    gen_ep   = []
    disc_ep  = []
    class_ep = []
    bus_ep   = []
    mode_ep  = []
    import time
    start_time = time.time()
    for epoch in range(Num_epochs):
        batch_idx = 1
        gen_loss_ep = 0
        disc_loss_ep = 0
        class_loss_ep = 0
        bus_loss_ep = 0
        mode_loss_ep = 0
        for real,labels,one_hot in data_loader:
            real = real.to(device)
            labels = labels.to(device)
            labels1 = torch.tensor([ft.get_word_vector(word) for word in one_hot]).to(device)
            one_hot_encoded = custom_one_hot_encode(one_hot, class_mapping)
            #Disc training
            temp = 0
            for _ in range(critic_iterations):
                noise = torch.randn((real.shape[0],z_dim,1,1)).to(device)
                fake = gen(noise,labels1)
                disc_labels = labels.view(real.shape[0],6,7,7).to(device)
                critic_real = critic(real,disc_labels).reshape(-1)
                critic_fake = critic(fake,disc_labels).reshape(-1)
                gp = gradient_penalty(critic,disc_labels,real,fake,device=device)
                loss_critic = (
                    -(torch.mean(critic_real)-torch.mean(critic_fake))+lambda_gp*gp
                    )
                opt_disc.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_disc.step()
                temp += loss_critic.item()
            
            disc_loss_ep += temp/5
            noise = torch.randn((real.shape[0],z_dim,1,1)).to(device)
            fake = gen(noise,labels1).to(device)
            text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)
            combined = model1(fake).to(device)
            text_features = model.encode_text(text_inputs)
            combined = combined/combined.norm(dim=-1, keepdim=True)
            text_features = text_features/text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * combined @ text_features.T).to(device).softmax(dim=-1)
            one_hot_encoded = one_hot_encoded.to(device)
            ce_loss = criterion(similarity,one_hot_encoded)
            disc_labels = labels.view(real.shape[0],6,7,7).to(device)
            output = critic(fake,disc_labels).reshape(-1)
            loss_gen = -torch.mean(output)

            bus_loss = 0
            if epoch >= thres_ep:
                noise = torch.randn((50,z_dim,1,1)).to(device)
                bus_emb = torch.tensor(word_vectors[words.index(missing)]).to(device)
                bus_emb = bus_emb.unsqueeze(0).repeat(50, 1)
                gen_bus = gen(noise,bus_emb)
                gen_bus = gen_bus.to(torch.float16)
                combined = model.visual.attnpool(gen_bus)
                class_lab = [missing]
                one_hot_encoded = torch.tensor([1.0,0,0,0,0,0,0,0],dtype=torch.float16,device=device)
                one_hot_encoded = one_hot_encoded.repeat(50, 1)
                text_inputs1 = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes1]).to(device)
                text_features = model.encode_text(text_inputs1)
                combined = combined/combined.norm(dim=-1, keepdim=True)
                text_features = text_features/text_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * combined @ text_features.T).to(device).softmax(dim=-1)
                one_hot_encoded = one_hot_encoded.to(device)
                bus_loss = criterion(similarity,one_hot_encoded)

            #mode loss
            mode_loss = 0
            if epoch <= thres_ep:
                noise = torch.randn((1,z_dim,1,1)).to(device)
                x = random.randint(0, 6)
                y = random.choice([i for i in range(0,6) if i not in [x]])
                x_emb = torch.tensor(word_vectors[words.index(classes[x])]).unsqueeze(0).to(device)
                y_emb = torch.tensor(word_vectors[words.index(classes[y])]).unsqueeze(0).to(device)
                X_one_hot_encoded = custom_one_hot_encode([classes[x]], class_mapping)
                y_one_hot_encoded = custom_one_hot_encode([classes[y]], class_mapping)
                x_out = gen(noise,x_emb).to(device)
                y_out = gen(noise,y_emb).to(device)
                zx = x_emb.unsqueeze(2).unsqueeze(3)
                zy = y_emb.unsqueeze(2).unsqueeze(3)
                zx = torch.cat((noise,zx),dim=1)
                zy = torch.cat((noise,zy),dim=1)
                mode_loss = torch.mean(torch.abs(y_out - x_out)) / torch.mean(
                    torch.abs(zy - zx))
                eps = 1 * 1e-5
                mode_loss = 1 / (mode_loss + eps)
            else:
                noise = torch.randn((1,z_dim,1,1)).to(device)
                x = random.randint(0, 7)
                y = random.choice([i for i in range(0,7) if i not in [x]])
                x_emb = torch.tensor(word_vectors[words.index(classes[x])]).unsqueeze(0).to(device)
                y_emb = torch.tensor(word_vectors[words.index(classes[y])]).unsqueeze(0).to(device)
                X_one_hot_encoded = custom_one_hot_encode([classes[x]], class_mapping)
                y_one_hot_encoded = custom_one_hot_encode([classes[y]], class_mapping)
                x_out = gen(noise,x_emb).to(device)
                y_out = gen(noise,y_emb).to(device)
                zx = x_emb.unsqueeze(2).unsqueeze(3)
                zy = y_emb.unsqueeze(2).unsqueeze(3)
                zx = torch.cat((noise,zx),dim=1)
                zy = torch.cat((noise,zy),dim=1)
                mode_loss = torch.mean(torch.abs(y_out - x_out)) / torch.mean(
                    torch.abs(zy - zx))
                eps = 1 * 1e-5
                mode_loss = 1 / (mode_loss + eps)

            final_loss = loss_gen + lambda_ce*ce_loss + lambda_mode*mode_loss + lambda_bus*bus_loss
            opt_gen.zero_grad()
            final_loss.backward()
            opt_gen.step()

            gen_loss_ep += loss_gen.item()
            class_loss_ep += ce_loss.item()
            mode_loss_ep += mode_loss.item()
            if epoch >= thres_ep:
                bus_loss_ep += bus_loss.item()
            elapsed_time = time.time() - start_time

            if batch_idx%10 == 0:
                print('----------------------------------------------------------------------------------------------------',file = op_file,flush=True)
                print('----------------------------------------------------------------------------------------------------')
                progress = ((epoch)*len(data_loader) + batch_idx)/(len(data_loader)*Num_epochs)
                eta = elapsed_time / (progress + 1e-5) * (1 - progress)
                time_format = time.strftime("%H:%M:%S", time.gmtime(eta))
                t = time.strftime("%H:%M:%S", time.localtime())
                print(
                    f"[{t}] Epoch [{epoch}/{Num_epochs-1}],Batch [{batch_idx}/{len(data_loader)}]    ETA: {time_format} \nCritic loss         :{loss_critic:.4f} \nGenerator loss      :{loss_gen:.4f}",file = op_file,flush=True
                )
                print(
                    f"Discriminatory loss :{ce_loss}",file = op_file,flush=True
                )
                if epoch >= thres_ep:
                    print(
                    f"bus loss:           :{bus_loss}",file = op_file,flush=True
                    )
                print(
                    f"Mode loss           :{mode_loss}",file = op_file,flush=True
                )
                print(
                    f"[{t}] Epoch [{epoch}/{Num_epochs-1}],Batch [{batch_idx}/{len(data_loader)}]    ETA:{time_format} \nCritic loss         :{loss_critic:.4f} \nGenerator loss      :{loss_gen:.4f}"
                )
                print(
                    f"Discriminatory loss :{ce_loss}"
                )
                if epoch >= thres_ep:
                    print(
                    f"bus loss:           :{bus_loss}"
                    )
                print(
                    f"Mode loss           :{mode_loss}"
                )
            batch_idx += 1
        gen_ep.append(gen_loss_ep)
        disc_ep.append(disc_loss_ep)
        class_ep.append(class_loss_ep)
        mode_ep.append(mode_loss_ep)
        if epoch >= thres_ep:
            bus_ep.append(bus_loss_ep)
        print('----------------------------------------------------------------------------------------------------',file = op_file,flush=True)
        print('----------------------------------------------------------------------------------------------------')
        print(f"For epoch {epoch}:")
        print(f"For epoch {epoch}:",file = op_file,flush=True)
        print(f"disc loss           : {disc_loss_ep}",file = op_file,flush=True)
        print(f"gen loss            : {gen_loss_ep}",file = op_file,flush=True)
        print(f"classification loss : {class_loss_ep}",file = op_file,flush=True)
        print(f"disc loss           : {disc_loss_ep}")
        print(f"gen loss            : {gen_loss_ep}")
        print(f"classification loss : {class_loss_ep}")
        if epoch >= thres_ep:
            print(f"bus loss            : {bus_loss_ep}")
            print(f"bus loss            : {bus_loss_ep}",file = op_file,flush=True)
            if bus_loss_ep <= bus_thresh:
                print("Early stopping...")
                print("Early stopping...",file = op_file,flush=True)
                if check == False:
                    torch.save(gen.state_dict(), 'generator_model_mid.pth')
                    check = True
    print('Saving models and loss histories',file = op_file,flush=True)
    print('Saving models and loss histories')
with open('disc_loss.pkl', 'wb') as file:
    pickle.dump(disc_ep, file)
with open('gen_loss.pkl', 'wb') as file:
    pickle.dump(gen_ep, file)
with open('class_loss.pkl', 'wb') as file:
    pickle.dump(class_ep, file)
with open('bus_loss.pkl', 'wb') as file:
    pickle.dump(bus_ep, file)
with open('mode_loss.pkl', 'wb') as file:
    pickle.dump(mode_ep, file)
torch.save(critic.state_dict(), 'critic_model.pth')
torch.save(gen.state_dict(), 'generator_model.pth')
