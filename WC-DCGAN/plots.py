import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('gen_loss.pkl', 'rb') as file:
    gen_loss = pickle.load(file)
with open('class_loss.pkl', 'rb') as file:
    class_loss = pickle.load(file)    
with open('disc_loss.pkl', 'rb') as file:
    disc_loss = pickle.load(file)   
with open('bus_loss.pkl', 'rb') as file:
    bus_loss = pickle.load(file) 
with open('mode_loss.pkl', 'rb') as file:
    mode_loss = pickle.load(file) 
with open('retrain_loss.pkl', 'rb') as file:
    retrain_loss = pickle.load(file) 
gen_combined = []
for i in range(gen_loss.__len__()):
    gen_combined.append(gen_loss[i] + class_loss[i] + mode_loss[i])

losses = {'gen_loss' : [gen_loss,'Generator loss'],'class_loss' : [class_loss,'Classification loss']
          ,'disc_loss' : [disc_loss,'Critic loss'],'bus_loss' : [bus_loss,'Unseen class classification loss']
          ,'mode_loss' : [mode_loss,'Mode seeking loss'],'retrain_loss' : [retrain_loss,'Fine-tuning loss'],
          'gen_combined' : [gen_combined,'Generator combined loss']}

for loss in losses:
    plt.title(losses[loss][1])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    modified_loss = [num / losses[loss][0].__len__() for num in losses[loss][0]]
    plt.plot(np.arange(0,modified_loss.__len__(),1), modified_loss, color="red")
    plt.show()
    plt.savefig(loss)
    plt.close()


plt.title('WC-DCGAN loss')
modified_loss = [num / gen_loss.__len__() for num in gen_loss]
plt.plot(np.arange(0,modified_loss.__len__(),1), modified_loss, color="blue",label='Generator')
modified_loss = [num / disc_loss.__len__() for num in disc_loss]
plt.plot(np.arange(0,modified_loss.__len__(),1), modified_loss, color="red",label='Critic')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
plt.savefig('overall WC-DCGAN Loss')
plt.close()

