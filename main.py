import os
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 
from sklearn.metrics import ConfusionMatrixDisplay

from models.sstvit import SSTViT
from utils.dataloader import HSI_Dataset
from utils.metrics import output_metric
from utils.utils import load_hsi_mat_data
from train import train_epoch, test_epoch
from predict import predict_epoch

#-------------------------------------------------------------------------------
# setting the parameters
# model mode
mode = "train" # train or test
pretrained = False # pretrained or not
model_path = r"" # model path

# model settings
model_type = "SSTViT"
patches = 5
band_patches = 3
num_classes = 3

# training settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DP = True
gpu = "1,2,3"
epoch = 50
test_freq = 500
batch_size = 256
learning_rate = 5e-4
weight_decay = 0
gamma = 0.9
ignore_index = 0

# data settings
num_samples = 500 # number of training samples
HSI_data_path = r"/home/ljs/BCDMNet/data/China_Change_Dataset.mat"

# time setting
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# GPU settings
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
cudnn.deterministic = True
cudnn.benchmark = False
#-------------------------------------------------------------------------------

# make the run folder in logs
#-------------------------------------------------------------------------------
time_now = time.localtime()
time_folder = os.path.join("logs", time.strftime("%Y-%m-%d-%H-%M-%S", time_now))
os.makedirs(time_folder, exist_ok=True)

# Initialize TensorBoard writer
os.makedirs(os.path.join(time_folder, "SummaryWriter"), exist_ok=True)
writer = SummaryWriter(log_dir=os.path.join(time_folder, "SummaryWriter"))

# Log basic parameters
params = {
    "Model Type": model_type,
    "Epochs": epoch,
    "Batch Size": batch_size,
    "Learning Rate": learning_rate,
    "Weight Decay": weight_decay,
    "Gamma": gamma,
    "GPU(s)": gpu,
    "training samples": num_samples,
}

for key, value in params.items():
    writer.add_text(key, str(value), 0)  
#-------------------------------------------------------------------------------

# data loading
#-------------------------------------------------------------------------------
data_t1, data_t2, train_label, test_label, all_label = load_hsi_mat_data(HSI_data_path, num_samples)
plt.imsave(os.path.join(time_folder, "train_label.png"), train_label, cmap='gray_r', dpi=300)
plt.imsave(os.path.join(time_folder, "test_label.png"), test_label, cmap='gray_r', dpi=300)
plt.imsave(os.path.join(time_folder, "data_label.png"), all_label, cmap='gray_r', dpi=300)
train_label_image = plt.imread(os.path.join(time_folder, "train_label.png"))
test_label_image = plt.imread(os.path.join(time_folder, "test_label.png"))
data_label_image = plt.imread(os.path.join(time_folder, "data_label.png"))
writer.add_image('Train Label', train_label_image, 0, dataformats='HWC')
writer.add_image('Test Label', test_label_image, 0, dataformats='HWC')
writer.add_image('Data Label', data_label_image, 0, dataformats='HWC')

if data_t1.shape == data_t2.shape and len(data_t1.shape) == 3 and len(data_t2.shape) == 3:
    height, width, band = data_t1.shape
else:
    raise ValueError("data size error!")
#-------------------------------------------------------------------------------

# models
#-------------------------------------------------------------------------------
model = SSTViT(
    image_size = patches,
    near_band = band_patches,
    num_patches = band,
    num_classes = num_classes,
    dim = 32,
    depth = 2,
    heads = 4,
    dim_head = 16,
    mlp_dim = 8,
    b_dim = 512,
    b_depth = 3,
    b_heads = 8,
    b_dim_head= 32,
    b_mlp_head = 8,
    dropout = 0.2,
    emb_dropout = 0.1,
)
#-------------------------------------------------------------------------------

# obtain dataset and dataloader
#-------------------------------------------------------------------------------
train_dataset = HSI_Dataset(data_t1, data_t2, train_label, patches, band_patches, "train")
test_dataset = HSI_Dataset(data_t1, data_t2, test_label, patches, band_patches, "train")
all_dataset = HSI_Dataset(data_t1, data_t2, all_label, patches, band_patches, "test")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False)
#-------------------------------------------------------------------------------

# model settings
#-------------------------------------------------------------------------------
if DP:
    model = torch.nn.DataParallel(model)
model = model.to(device)
# criterion
criterion = nn.CrossEntropyLoss(ignore_index=ignore_index).to(device)
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# scheduler
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epoch//10, gamma=gamma)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch//10, eta_min=5e-4)
#-------------------------------------------------------------------------------

# start training or testing
#-------------------------------------------------------------------------------
if mode == "train":
    # if pretrained
    if pretrained:
        model.load_state_dict(torch.load(model_path))
        print("load model path : " + model_path)
    # train
    print("===============================================================================")
    print("start training")
    tic = time.time()

    for e in range(epoch): 
        model.train()
        train_acc, train_loss, label_t, prediction_t = train_epoch(model, train_loader, criterion, optimizer, e, epoch, device)
        scheduler.step()
        OA_train, AA_train, Kappa_train, CA_train, CM_train = output_metric(label_t, prediction_t, ignore_index=0) 
        print("Epoch: {:03d} | train_loss: {:.4f} | train_acc: {:.4f}%".format(e+1, train_loss, train_acc))
        
        # Logging to TensorBoard
        writer.add_scalar('Loss/train', train_loss, e+1)
        writer.add_scalar('Accuracy/train', train_acc, e+1)
        writer.add_scalar('AA/train', AA_train, e+1)  
        writer.add_scalar('Kappa/train', Kappa_train, e+1)  

        if ((e+1) % test_freq == 0) | (e == epoch - 1):
            print("===============================================================================")
            print("start testing")      
            model.eval()
            label_v, prediction_v = test_epoch(model, test_loader, criterion, device)
            OA_test, AA_test, Kappa_test, CA_test, CM_test = output_metric(label_v, prediction_v, ignore_index=0)
            if (e != epoch -1):
                print("Epoch: {:03d}  =>  OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(e+1, OA_test, AA_test, Kappa_test))
            
            # Logging test metrics to TensorBoard
            writer.add_scalar('Accuracy/test', OA_test, e+1)
            writer.add_scalar('AA/test', AA_test, e+1) 
            writer.add_scalar('Kappa/test', Kappa_test, e+1)  
            
            cm_display = ConfusionMatrixDisplay(CM_test)
            cm_display.plot(cmap='Blues', colorbar=False)
            plt.title('Confusion Matrix')
            plt.tight_layout()
            cm_image_path = os.path.join(time_folder, f"confusion_matrix_{e+1}.png")
            plt.savefig(cm_image_path, dpi=300)
            plt.close() 
            cm_image = plt.imread(cm_image_path)
            writer.add_image(f'Confusion Matrix/Epoch {e+1}', cm_image, e+1, dataformats='HWC')

            print("===============================================================================")

    toc = time.time()
    print("Running Time: {:.2f}".format(toc-tic))
    print("end training and testing")
    print("===============================================================================")
    print("Final result:")
    print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA_test, AA_test, Kappa_test))
    print("CA:", end="")
    print(CA_test)
    print("Confusion Matrix:")
    print(CM_test)
    print("===============================================================================")

elif mode == "test":
    model.load_state_dict(torch.load(model_path))
    print("load model path : " + model_path)
    print("===============================================================================")

else:
    raise ValueError("model error!")

if mode == "train":
    # save model and its parameters 
    torch.save(model, os.path.join(time_folder, "model.pt"))
    torch.save(model.state_dict(), os.path.join(time_folder, "model_state_dict.pth"))

print("start predicting")
model.eval()

# output classification maps
pre_u = predict_epoch(model, all_loader, device)
prediction = np.zeros((height, width), dtype=float)
for idx, (row, col) in enumerate(np.ndindex(height, width)):
    prediction[row, col] = pre_u[idx]  
plt.imsave(os.path.join(time_folder, "predict_result.png"), prediction, cmap='gray_r', dpi=300)

# write prediction image to TensorBoard
prediction_image = plt.imread(os.path.join(time_folder, "predict_result.png"))
writer.add_image('Prediction Result', prediction_image, 0, dataformats='HWC')

# close the TensorBoard writer
writer.close()

print("end predicting")
print("===============================================================================")

# tensorboard --logdir=/home/ljs/BCDMNet/logs/2024-10-22-04-31-09/SummaryWriter --port=6061
# ssh -NfL 8080:127.0.0.1:6061 ljs@172.18.206.54 -p 6522


