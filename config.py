# model and data configurations
target_size = (224, 224)
grid_size = tuple([dim // (2 ** 5) for dim in target_size])
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
num_classes = 21

# training configuration
batch_size = 16
epochs = 100
