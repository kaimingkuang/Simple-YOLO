# model and data configurations
target_size = (224, 224)
grid_size = tuple([dim // (2 ** 5) for dim in target_size])
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
num_classes = 20

# training configuration
batch_size = 64
epochs = 100
w_cls = 1
w_reg = 5
w_pos = 2
max_lr = 1e-3
min_lr = 1e-6
final_div_factor = min_lr / max_lr
pct_start = 0.05

# evaluation configuration
prob_thresh = 0.1
overlap_iou_thresh = 0.5
