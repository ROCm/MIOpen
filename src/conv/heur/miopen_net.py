#1/usr/bin/env python3
import pandas as pd
import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from model import MIOpenNet
from find_dataset import FindDataset
from gen_code import gen_metadata_cpp, process_onnx_file

random_state = 42

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def parse_args():
    # TODO: splilt the args into groups for each command 
    parser = argparse.ArgumentParser(description='MIOpenNet')
    parser.add_argument('command', type=str, choices=['meta', 'train', 'gen_cpp', 'gen_bin'], help='Perform function')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', type=str, default=None,
                        help='For Saving the current Model')
    parser.add_argument('--arch', type=str, default='all', choices = ['gfx906', 'gfx908', 'all'], help='Target Arch for the model')
    parser.add_argument('--dir', type=str, default=None, choices=[None, 'fwd', 'bwd', 'wrw'], help='Target direction for the model')
    parser.add_argument('--data_filename', type=str, required=True, help='Filename for data')
    parser.add_argument('--output_file', type=str, help='Filename to write to( appropriate extension will be added)')
    args = parser.parse_args()
    if args.arch == 'all':
        args.arch = ['gfx906', 'gfx908']
    else:
        args.arch = [args.arch]
    return args

def gen_meta(args):
    assert('pd' in args.data_filename)
    uber_df = pd.read_pickle(args.data_filename)
    uber_metadata = {}
    # add json extension if not there
    if os.path.isfile(args.output_file) and 'json' in args.output_file:
        with open(args.output_file) as jf:
            jdict = jf.read()
        uber_metadata = json.loads(jdict)
    # TODO: ignores the arch argument on the command line 
    for arch in  uber_df.arch.unique():
        if arch not in uber_metadata:
            metadata = {}
        else:
            metadata = uber_metadata[arch]
        full_dataset = FindDataset(args.data_filename, arch)
        df = uber_df[uber_df.arch == arch]
        # Feature names and solver map are arch specific
        metadata['feature_names'] = full_dataset.feature_names
        if 'mu' not in metadata:
            metadata['mu'] = {}
        if 'sigma' not in metadata:
            metadata['sigma'] = {}
        if 'solver_map' not in metadata:
            metadata['solver_map'] = {}
        for direction in df.direction.unique():
            # direction specific metadata
            dir_dataset = FindDataset(args.data_filename, arch, direction=direction)
            metadata['mu'][direction] = dir_dataset.mu.tolist()
            metadata['sigma'][direction] = dir_dataset.sigma.tolist()
            metadata['solver_map'][direction] = dir_dataset.cat_map
        uber_metadata[arch] = metadata # Overwrite the old one
    with open(args.output_file, 'w') as jf:
        jf.write(json.dumps(uber_metadata, sort_keys=True, indent=2))
    code = gen_metadata_cpp(uber_metadata)
    with open(args.output_file, 'w') as cpp:
        cpp.write(code)

def network(args):
    # Training settings
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    for arch in args.arch:
        if not args.dir:
          dirs = ['fwd', 'bwd', 'wrw']
        else:
          dirs = [args.dir]
        for direction in dirs:
          full_dataset = FindDataset(args.data_filename, arch, direction=direction)
          train_size = int(0.8 * len(full_dataset))
          test_size = len(full_dataset) - train_size
          train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
          train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
          test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

          model = MIOpenNet(full_dataset.num_features, full_dataset.num_solvers).to(device)
          optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

          scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
          for epoch in range(1, args.epochs + 1):
              train(args, model, device, train_loader, optimizer, epoch)
              test(model, device, test_loader)
              scheduler.step()

          if args.save_model:
              name = os.path.join(args.save_model, '-'.join([direction, arch]))
              # torch.save(model.state_dict(), name + ".pt")
              from torch.autograd import Variable
              x = Variable(torch.randn(1, full_dataset.num_features), requires_grad=True)
              torch_out = torch.onnx._export(model,             # model being run
                                     x,                       # model input (or a tuple for multiple inputs)
                                     name + ".onnx",       # where to save the model (can be a file or file-like object)
                                     export_params=True)      # store the trained parameter weights inside the model file
              const_filename, graph_obj = process_onnx_file(name + ".onnx", name, direction, arch)
def gen_cpp(args):
    metadata = {}
    assert('json' in args.data_filename)
    with open(args.data_filename) as jf:
        metadata = json.loads(jf.read())
    code = gen_metadata_cpp(metadata)
    with open(args.output_file + '.cpp', 'w') as cpp:
        cpp.write(code)

def gen_bin(args):
    assert('onnx' in args.data_filename)
    assert(args.dir is not None)
    assert(args.arch is not None and len(args.arch) > 0)
    const_filename, graph_obj = process_onnx_file(args.data_filename, args.output_file, args.dir, args.arch[0])


def main():
    args = parse_args()
    if args.command == 'meta':
        gen_meta(args)
    elif args.command == 'train':
        network(args)
    elif args.command == 'gen_cpp':
        gen_cpp(args)
    elif args.command == 'gen_bin':
        gen_bin(args)

if __name__=='__main__':
  main()

