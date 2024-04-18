import torch
import time
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from p_data import *
from p_model import *
from p_utils import *

#parameters settings
output_folder = 'outputs/gen'
batch_size = 1
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")

#model
model = Generator().to(device)

#dataset, dataloader
test_dataset = ImageDataset("Pic_test", ratio=0.25)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

if __name__ == "__main__":
    #load checkpoint
    load_checkpoint(device, 'outputs/checkpoints/model_2000.pt', model)

    #testing
    model.eval()
    print("Testing...")
    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), ncols=144):
        #batch data
        z_M = batch['g_input'].type(torch.float).to(device)

        #forward - generate
        gen_img = model(z_M)
        img_grid = denormalize(gen_img)
        save_image(img_grid, output_folder + '/' + f"{i+1}.png", nrow=1, normalize=False)
    print('TESTING COMPLETE')