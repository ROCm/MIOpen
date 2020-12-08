#!/usr/bin/env python3

import argparse
from re import search
import pandas as pd
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Parse Find DB and generate classification dataset')
    parser.add_argument('filename', type=str, help='File to parse')
    parser.add_argument('out_filename', type=str, help='Output filename')
    args = parser.parse_args()
    args.filename = get_file_list(args.filename)
    return args

# Code copied from Tuna's Prasing module
# TODO: create Tuna packages so we can export this logic without copying
invers_dir_map = {'F':'fwd', 'B':'bwd', 'W':'wrw'}
fds_3d = ['pad_d', 'pad_h', 'pad_w',
          'out_channels',
          'fil_d', 'fil_w', 'fil_h',
          'dilation_d', 'dilation_w', 'dilation_h',
          'conv_stride_d', 'conv_stride_w', 'conv_stride_h',
          'in_channels',
          'in_d', 'in_w', 'in_h',
          'batchsize', 'group_count']

fds_2d = ['pad_h', 'pad_w',
          'out_channels',
          'fil_w', 'fil_h',
          'dilation_w', 'dilation_h',
          'conv_stride_w', 'conv_stride_h',
          'in_channels',
          'in_w', 'in_h',
          'batchsize', 'group_count']

def get_file_list(filename):
    files = []
    if os.path.isdir(filename):
        all_files = os.listdir(filename)
        for name in all_files:
            if not name.startswith('gfx'):
                continue
            if 'ufdb' in name:
                continue
            if not 'fdb' in name:
                continue
            lst = name.split('.')
            name = os.path.join(filename, name)
            files.append(name)
    else:
        files = [filename]
    return files
def parse_pdb_key(key):
    pattern_3d = '[0-9]x[0-9]x[0-9]'
    group_count = '1'
    if key.find('_') != -1:
        optional_prt = key[key.find('_')+1:]
        key = key[:key.find('_')]
        if optional_prt[0] != 'g':
            raise ValueError('Only group count in optional part is supported')
        group_count = optional_prt[1:]
        if not group_count.isdigit():
            raise ValueError('Group count has to be integer')

    if search(pattern_3d, key):
        vals, precision, direction = parse_3d(key, group_count)
        fds = fds_3d
    else:
        vals, precision, direction = parse_2d(key, group_count)
        fds = fds_2d

    vals2 = []
    for v in vals:
        if v.isdigit(): v = int(v)
        vals2.append(v)

    return fds, vals2, precision, invers_dir_map[direction]

def parse_2d(key, group_count):
    tmp = key.split('-')
    direction = tmp[14]
    if direction == 'F':
        in_channels = tmp[0]
        in_h = tmp[1]
        in_w = tmp[2]
        out_channels = tmp[4]
        # tmp[5], tmp[6]:  outputsize is ignored in db
    else:
        out_channels = tmp[0]
        in_h = tmp[5]
        in_w = tmp[6]
        in_channels = tmp[4]

    fil_h, fil_w = tmp[3].split('x')
    batchsize = tmp[7]
    pad_h, pad_w = tmp[8].split('x')
    conv_stride_w, conv_stride_h = tmp[9].split('x')
    dilation_h, dilation_w = tmp[10].split('x')
    # unused bias = tmp[11]
    # unused layout = tmp[12]
    precision = tmp[13]

    vals_2d = [pad_h, pad_w,
               out_channels,
               fil_w, fil_h,
               dilation_w, dilation_h,
               conv_stride_w, conv_stride_h,
               in_channels,
               in_w, in_h,
               batchsize,
               group_count]

    return vals_2d, precision, direction

def parse_3d(key, group_count):
    #sample 3D
    #256-16-56-56-1x1x1-64-16-56-56-4-0x0x0-1x1x1-1x1x1-0-NCHW-FP32-F=
    tmp = key.split('-')
    direction = tmp[16]
    if direction == 'F':
        in_channels = tmp[0]
        in_d = tmp[1]
        in_h = tmp[2]
        in_w = tmp[3]
        out_channels = tmp[5]
        # tmp[5], tmp[6]:  outputsize is ignored in db
    else:
        out_channels = tmp[0]
        in_d = tmp[6]
        in_h = tmp[7]
        in_w = tmp[8]
        in_channels = tmp[5]

    fil_d, fil_h, fil_w = tmp[4].split('x')
    batchsize = tmp[9]
    pad_d, pad_h, pad_w = tmp[10].split('x')
    conv_stride_d, conv_stride_w, conv_stride_h = tmp[11].split('x')
    dilation_d, dilation_h, dilation_w = tmp[12].split('x')
    # unused bias = tmp[13]
    # unused layout = tmp[14]
    precision = tmp[15]

    vals_3d = [pad_d, pad_h, pad_w,
               out_channels,
               fil_d, fil_w, fil_h,
               dilation_d, dilation_w, dilation_h,
               conv_stride_d, conv_stride_w, conv_stride_h,
               in_channels,
               in_d, in_w, in_h,
               batchsize,
               group_count]

    return vals_3d, precision, direction
def main():
    args = parse_args()
    all_params = {}
    all_fds = list(set(fds_2d + fds_3d)) + ['arch', 'num_cu', 'backend', 'precision', 'direction']
    for sol_idx in range(5): # assuming a max of 5 solvers per config
        s_idx = str(sol_idx)
        all_fds.append('algo_' + s_idx)
        all_fds.append('sol_name_' + s_idx)
        all_fds.append('sol_runtime_' + s_idx)
        all_fds.append('sol_ws_sz_' + s_idx)
    for fd in all_fds:
        all_params[fd] = []
    
    for filename in args.filename:
        print('Processing file: {}'.format(filename))
        find_db_name = os.path.basename(filename)
        lst = find_db_name.split('.')
        assert len(lst) == 4
        assert lst[2] == 'fdb'
        assert lst[3] == 'txt'
        backend = lst[1]
        if '_' in lst[0]:
          arch, num_cu = lst[0].split('_')
        else:
          arch = 'gfx908'
          num_cu = '78'
        with open(filename) as fdb_file
          for line in fdb_file:
              key, val = line.split('=')
              fds, fd_vals, precision, direction = parse_pdb_key(key)
              sol_params = {k:v for k,v in zip(fds, fd_vals)}
              sol_params['precision'] = precision
              sol_params['direction'] = direction
              sol_params['arch'] = arch
              sol_params['num_cu'] = num_cu
              sol_params['backend'] = backend
              # sort the algorithms by kernel_time
              algo_list = []
              for sol_idx, kinder in enumerate(val.split(';')):
                algo_map = {}
                algo, params = kinder.split(':')
                algo_map['algo'] = algo
                sol_name, runtime, ws_sz, _, _ = params.split(',')
                algo_map['sol_name'] = sol_name
                algo_map['sol_runtime'] = runtime
                algo_map['sol_ws_sz'] = ws_sz
                algo_list.append(algo_map)
              algo_list = sorted(algo_list, key = lambda x: x['sol_runtime'])
              for idx, algo in enumerate(algo_list):
                idx = '_' + str(idx)
                for key, val in algo.items():
                  sol_params[key + idx] = val

              for k in all_fds:
                  if k in sol_params.keys():
                      all_params[k].append(sol_params[k])
                  else:
                      all_params[k].append(None)

    df = pd.DataFrame(all_params)
    df = df.dropna(how='all', axis=1);
    df['is_gemm'] = (df.algo_0 == 'miopenConvolutionFwdAlgoGEMM')
    df.to_pickle(args.out_filename)

if __name__=='__main__':
    main()
