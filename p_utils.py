import torch
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

#save model for resume training
save_model_path = 'project/Boundless-in-Pytorch-master/outputs/'
def save_model(epochs_trained, model_G, model_D, optimizer_G, optimizer_D, plot_list):
    torch.save({
        'epoch': epochs_trained,
        'model_G_state_dict': model_G.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'model_D_state_dict': model_D.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'plot_list': plot_list
    }, save_model_path + f'checkpoints/model_{epochs_trained}.pt')


#saving loss, metric plots after complete training
def save_plots(fig, train_loss_pixel, train_loss_adv, train_loss_G, train_loss_D):
    #pixel, adv loss plots
    ax1 = fig.get_axes()[0]
    ax2 = fig.get_axes()[2]

    ax1.plot(train_loss_pixel, color='orange', linestyle='-', label='real score')
    ax1.tick_params('y', color='orange')

    ax2.plot(train_loss_adv, color='red', linestyle='-', label='fake score')
    ax2.set(xlabel='Epochs')
    ax1.tick_params('y', color='red')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)

    #G, D loss plots
    ax1 = fig.get_axes()[1]
    ax2 = fig.get_axes()[3]

    ax1.plot(train_loss_G, color='green', linestyle='-', label='train G loss')
    ax1.tick_params('y', color='green')

    ax2.plot(train_loss_D, color='blue', linestyle='-', label='train D loss')
    ax2.set(xlabel='Epochs')
    ax2.tick_params('y', color='blue')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)

    #save figure
    plt.savefig(save_model_path + 'loss.png')
    fig.get_axes()[0].cla()
    fig.get_axes()[1].cla()
    fig.get_axes()[2].cla()
    fig.get_axes()[3].cla()


#checkpoint (overrides lr, scheduler settings)
def load_checkpoint(device, pt_path, model_G, model_D=None, optimizer_G=None, optimizer_D=None):
    checkpoint = torch.load(pt_path, map_location=device)
    model_G.load_state_dict(checkpoint['model_G_state_dict'])
    if model_D and optimizer_G and optimizer_D:
        model_D.load_state_dict(checkpoint['model_D_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    start_epoch = checkpoint['epoch']+1
    plot_list = checkpoint['plot_list']
    return start_epoch, plot_list


#seed function
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True