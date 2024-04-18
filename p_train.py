import os
import torch
import time
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from p_data import *
from p_model import *
from p_utils import *

##name and SID
name = "CHENG Chi Yin"
SID = "1155160221"

#parameters settings
end_epoch = 10000
lr_G = 1e-4
lr_D = 1e-3
eps = 1e-8
batch_size = 12
accumulate_steps = 16 #effective batch = 256
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")

torch.backends.cudnn.benchmark = True

#model (generator, discriminator, extractor)
generator = Generator().to(device)
discriminator = Discriminator().to(device)
extractor = InceptionExtractor().to(device)
extractor.eval()

#dataset, dataloader
train_dataset = ImageDataset("project/Boundless-in-Pytorch-master/img_train", ratio=0.25)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_dataset = ImageDataset("project/Boundless-in-Pytorch-master/img_test", ratio=0.25)
test_loader = DataLoader(test_dataset, batch_size=9, shuffle=False, num_workers=4, pin_memory=True)

#criterion, optimizer
criterion_rec = torch.nn.L1Loss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_G, eps=eps, betas=(0.5, 0.9))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_D, eps=eps, betas=(0.5, 0.9))

#train step
def train_step(train_loader, model_G, model_D, criterion_rec, optimizer_G, optimizer_D):
    train_running = [0.0, 0.0, 0.0, 0.0] #pred_real, pred_fake, loss_G, loss_D
    accumulate_steps_last = len(train_loader)%accumulate_steps
    #set model to train mode
    model_G.train()
    model_D.train()
    print("Training...")
    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), ncols=144):
        #batch data
        x = batch['real_img'].type(torch.float).to(device)
        z = batch['masked_img'].type(torch.float).to(device)
        M = batch['mask'].type(torch.float).to(device)
        z_M = batch['g_input'].type(torch.float).to(device)
        class_cond = extractor(x).detach()

        # ------------------
        #  Train Generator
        # ------------------

        #forward - generate
        gen_img = model_G(z_M)

        #compute reconstruction loss
        loss_rec = criterion_rec(gen_img, x) 
        #compute adversarial loss
        x_hat = gen_img * M + z
        pred_fake = model_D(x_hat, M, class_cond)
        loss_adv = -pred_fake.mean()

        #compute generator total loss, scale loss
        loss_G = 1e-2 * loss_adv + loss_rec
        train_running[2] += loss_G.item()
        if ((i+1) > (len(train_loader)-accumulate_steps_last)): #last accumulate step
            loss_G = loss_G / accumulate_steps_last
        else:
            loss_G = loss_G / accumulate_steps
       
        #backward - compute gradients
        loss_G.backward()
        #backward - accumulate gradients for every X steps
        if (((i+1) % accumulate_steps) == 0) or ((i+1) == len(train_loader)):
            #update parameters, zero out gradients
            optimizer_G.step()
            optimizer_G.zero_grad()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        #forward - discriminate
        pred_real = model_D(x, M, class_cond)
        pred_fake = model_D(x_hat.detach(), M, class_cond)

        train_running[0] += (pred_real.mean()).item()
        train_running[1] += (pred_fake.mean()).item()

        #compute discriminator total loss, scale loss
        loss_D = torch.nn.functional.relu((1.0 - pred_real).mean()) + torch.nn.functional.relu((1.0 + pred_fake).mean())
        train_running[3] += loss_D.item()
        if ((i+1) > (len(train_loader)-accumulate_steps_last)): #last accumulate step
            loss_D = loss_D / accumulate_steps_last
        else:
            loss_D = loss_D / accumulate_steps

        #backward - compute gradients
        loss_D.backward()
        #backward - accumulate gradients for every X steps
        if (((i+1) % accumulate_steps) == 0) or ((i+1) == len(train_loader)):
            #update parameters, zero out gradients
            optimizer_D.step()
            optimizer_D.zero_grad()   
    epoch_loss = np.divide(train_running, i)
    img_grid = denormalize(torch.cat((z, gen_img, x_hat, x), -1))
    
    return epoch_loss, img_grid

#test step
def test_step(test_loader, model_G, epoch):
    test_path = 'project/Boundless-in-Pytorch-master/outputs/test/'
    """test_path = test_path + str(epoch+1) + '/'
    if not os.path.exists(test_path):
        os.makedirs(test_path)"""

    #testing
    model_G.eval()
    print("Testing...")
    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), ncols=144):
        #batch data
        z_M = batch['g_input'].type(torch.float).to(device)

        #forward - generate
        gen_img = model_G(z_M)
        img_grid = denormalize(gen_img)
        save_image(img_grid, test_path + f"{epoch+1}.png", nrow=3, normalize=False)
    print('TESTING COMPLETE')


if __name__ == "__main__":
    save_n_model = 10
    save_n_img = 1

    #create figure
    fig, ax = plt.subplots(1, 2, figsize = (20, 7), sharey=True)
    ax[0].twinx()
    ax[1].twinx()

    #load checkpoint
    resume = True
    if resume:
        start_epoch, plot_list = load_checkpoint(device, 'project/Boundless-in-Pytorch-master/outputs/checkpoints/model_80.pt', generator, discriminator, optimizer_G, optimizer_D)
        train_loss_pixel, train_loss_adv = plot_list[0], plot_list[1]
        train_loss_G, train_loss_D = plot_list[2], plot_list[3]
        plot_list = [train_loss_pixel, train_loss_adv, train_loss_G, train_loss_D]
        print(f"RESUME TRAINING")
    else:
        #setup_seed(42)
        start_epoch = 1
        train_loss_pixel, train_loss_adv, train_loss_G, train_loss_D = [], [], [], []
        plot_list = [train_loss_pixel, train_loss_adv, train_loss_G, train_loss_D]

    #training
    for epoch in range(start_epoch-1, end_epoch):
        print(f"[INFO]: Epoch {epoch+1} of {end_epoch}")
        train_epoch_loss, img_grid = train_step(train_loader, generator, discriminator, criterion_rec, optimizer_G, optimizer_D)

        #print stats
        train_loss_pixel.append(train_epoch_loss[0])
        train_loss_adv.append(train_epoch_loss[1])
        train_loss_G.append(train_epoch_loss[2])
        train_loss_D.append(train_epoch_loss[3])
        print(f"Real score: {train_epoch_loss[0]:.3f}, fake score: {train_epoch_loss[1]:.3f}")
        print(f"Training G loss: {train_epoch_loss[2]:.3f}, training D loss: {train_epoch_loss[3]:.3f}")

        #save model for every x epoch
        if ((epoch+1) % save_n_model) == 0:
            save_model(epoch+1, generator, discriminator, optimizer_G, optimizer_D, plot_list)

        #save loss, accuracy plots
        save_plots(fig, train_loss_pixel, train_loss_adv, train_loss_G, train_loss_D)

        #save image/test for every x epoch
        if ((epoch+1) % save_n_img) == 0:
            save_image(img_grid, 'project/Boundless-in-Pytorch-master/outputs/train/' + f"{epoch+1}.png", nrow=1, normalize=False)
            test_step(test_loader, generator, epoch)

        print('-'*50)
        
    print('TRAINING COMPLETE')