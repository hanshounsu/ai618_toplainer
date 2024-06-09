import os 
import torch 
import torch.nn as nn

# def set_device(x, device):
#     use_cuda = False
#     multi_gpu = False
#     if len(device) == 1 and device[0] > 0:
#         use_cuda = True 
#     elif len(device) > 1:
#         use_cuda = True 
#         multi_gpu = True 

#     # When input is tensor 
#     if isinstance(x, torch.Tensor): 
#         if use_cuda:
#             x = x.cuda(device[0] - 1)
#         else:
#             x = x.cpu()
#      # When input is model
#     elif isinstance(x, nn.Module): 
#         if use_cuda:
#             if multi_gpu:
#                 devices = [i - 1 for i in device]
#                 torch.cuda.set_device(devices[0])
#                 if isinstance(x, nn.DataParallel) and x.device_ids != devices:
#                     x = x.module
#                 if not isinstance(x, nn.DataParallel):
#                     x = nn.DataParallel(x, device_ids=devices).cuda()
#             else: 
#                 torch.cuda.set_device(device[0] - 1)
#                 x.cuda(device[0] - 1)
#         else: 
#             x.cpu()
#     # When input is tuple 
#     elif type(x) is tuple or type(x) is list:
#         x = list(x)
#         for i in range(len(x)):
#             x[i] = set_device(x[i], device)
#         x = tuple(x) 

#     return x 

def set_device(x, device):
    use_cuda = False
    multi_gpu = False
    if device != 'cpu':
        use_cuda = True 
    # elif len(device) > 1:
    #     use_cuda = True 
    #     multi_gpu = True 

    # When input is tensor 
    if isinstance(x, torch.Tensor): 
        if use_cuda:
            x = x.cuda(device)
        else:
            x = x.cpu()
     # When input is model
    elif isinstance(x, nn.Module): 
        if use_cuda:
            if multi_gpu:
                devices = [i - 1 for i in device]
                torch.cuda.set_device(devices[0])
                if isinstance(x, nn.DataParallel) and x.device_ids != devices:
                    x = x.module
                if not isinstance(x, nn.DataParallel):
                    x = nn.DataParallel(x, device_ids=devices).cuda()
            else: 
                # torch.cuda.set_device(torch.device(device))
                x.cuda(device)
        else: 
            x.cpu()
    # When input is tuple 
    elif type(x) is tuple or type(x) is list:
        x = list(x)
        for i in range(len(x)):
            x[i] = set_device(x[i], device)
        x = tuple(x) 

    return x 

def from_parallel(state_dict):
    from_parallel = False
    for key, _ in state_dict.items():
        if key.find('module.') != -1:
            from_parallel = True
            break 

    return from_parallel

def unwrap_parallel(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value

    return new_state_dict


def save_checkpoint(checkpoint_path, model, optimizer=None, learning_rate=None, iteration=None, verbose=False):
    checkpoint = {'state_dict': model.state_dict()}
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer._optimizer.state_dict()
    if learning_rate is not None:
        checkpoint['learning_rate'] = learning_rate
    if iteration is not None:
        checkpoint['iteration'] = iteration
    
    torch.save(checkpoint, checkpoint_path)

    if verbose: 
        print("Saving checkpoint to %s" % (checkpoint_path))

def load_checkpoint(checkpoint_path, model, optimizer=None, verbose=False):
    assert os.path.isfile(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model'].state_dict()
    else:
        raise AssertionError("No model weight found in checkpoint, %s" % (checkpoint_path))

    if from_parallel: 
        state_dict = unwrap_parallel(state_dict)

    model.load_state_dict(state_dict)

    objects = [model]
    if 'optimizer' in checkpoint and optimizer is not None:
        optimizer._optimizer.load_state_dict(checkpoint['optimizer'])
        objects.append(optimizer)
    if 'learning_rate' in checkpoint:
        learning_rate = checkpoint['learning_rate']
        objects.append(learning_rate)
    if 'iteration' in checkpoint:
        iteration = checkpoint['iteration']
        objects.append(iteration)

    if verbose:
        print("Loaded checkpoint from %s" % (checkpoint_path))

    if len(objects) == 1:
        objects = objects[0]

    return objects

class AverageMeter(object):
    def __init__(self):
        self.steps = 0
        self.total_num = 0
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.num = 0
        self.avg = 0.0

    def step(self, val, num=1):
        self.val = val
        self.sum += num*val
        self.num += num
        self.steps += 1
        self.total_num += num
        self.avg = self.sum/self.num