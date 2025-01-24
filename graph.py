''' Visualize training results '''
import matplotlib.pyplot as plt
import torch 

device = torch.device("cpu")
weights_pathj = 'dataset_crop2/output_weights/si/last.pth'
# weights_pathc = 'dataset_building_fixed_rawImg/output_weights/si/last.pth'
state_dict = torch.load(weights_pathj, map_location=device)
task_name = weights_pathj.split('/')[0].split('_')[1]
mode = weights_pathj.split('/')[2][:2]

train_loss = state_dict['train_loss']
val_loss = state_dict['val_loss']
f1_score = state_dict['val_score']
epochs = state_dict['epoch']

min_train_loss = min(train_loss)
min_val_loss = min(val_loss)
max_f1_score = max(f1_score)

min_train_epoch = epochs[train_loss.index(min_train_loss)]
min_val_epoch = epochs[val_loss.index(min_val_loss)]
max_f1_epoch = epochs[f1_score.index(max_f1_score)]

print(epochs[-1])
print(train_loss)
print(min_val_loss)
print(f1_score)


# print(state_dict['scaler_state_dict'])
# # print(state_dict['optimizer_state_dict']['state'])
# print(state_dict['time - eval_time'])
# print(state_dict['optimizer.param_groups[0][lr]'])

if len(train_loss) == len(val_loss) == len(f1_score):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Train Loss', color='blue')
    # plt.plot(epochs, val_loss, label='Validation Loss', color='orange')
    plt.plot(epochs, f1_score, label='F1 Score', color='green')
    
    # Đánh dấu các giá trị min/max
    plt.scatter(min_train_epoch, min_train_loss, color='darkblue', label='Min Train Loss', zorder=5)
    plt.text(min_train_epoch, min_train_loss, f'{min_train_loss:.4f}', color='darkblue')

    # plt.scatter(min_val_epoch, min_val_loss, color='darkorange', label='Min Val Loss', zorder=5)
    # plt.text(min_val_epoch, min_val_loss, f'{min_val_loss:.4f}', color='darkorange')

    plt.scatter(max_f1_epoch, max_f1_score, color='darkgreen', label='Max F1 Score', zorder=5)
    plt.text(max_f1_epoch, max_f1_score, f'{max_f1_score:.4f}', color='darkgreen')

    plt.xlabel('Epochs')
    plt.ylabel('Values')
    plt.title(f'({task_name}) Aerial_SwinB_{mode}, {epochs[-1]} epochs')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 5)
    plt.show()
    
    # gradients = []

    # # Sau mỗi bước tối ưu (optimizer.step)
    # for name, param in state_dict.named_parameters():
    #     if param.requires_grad and param.grad is not None:
    #         gradients.append(param.grad.norm().item())

    # # Vẽ biểu đồ
    # import matplotlib.pyplot as plt
    # plt.plot(gradients)
    # plt.xlabel('Parameter Index')
    # plt.ylabel('Gradient Norm')
    # plt.title('Gradient Norms Across Layers')
    # plt.show()

else:
    print("fail")