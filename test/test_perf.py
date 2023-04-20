#!/usr/bin/env python3
###############################################################################
#
# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################
"""Performance tracking script"""
import sys
import os
import re
import subprocess
import argparse
import csv
from decimal import Decimal

results_path = f"{os.path.dirname(__file__)}/perf_results"
TOLERANCE = -5  #tolerance 5%


def parse_args():
  """Function to parse cmd line arguments"""
  parser = argparse.ArgumentParser()
  group1 = parser.add_mutually_exclusive_group()

  parser.add_argument('--filename',
                      dest='filename',
                      type=str,
                      required=True,
                      help='Specify filename for model')
  group1.add_argument('--install_path',
                      dest='install_path',
                      help='Specify MIOpenDriver install path')
  group1.add_argument('--compare_results',
                      dest='compare_results',
                      action='store_true',
                      help='Compare current perf results to old perf results')
  parser.add_argument('--old_results_path',
                      dest='old_results_path',
                      type=str,
                      help='Specify full path to old results directory')
  parser.add_argument('--override',
                      dest='override',
                      type=str,
                      help='Specify driver cmd env vars')
  args = parser.parse_args()

  if args.compare_results and not args.old_results_path:
    parser.error(
        '--old_results_path and --compare_results must both be specified')

  return args


def run_driver_cmds(filename, install_path, override=None):
  """Parse model file and launch Driver cmds"""
  resfile = f"{results_path}/{filename}"
  model_path = f"{install_path}/share/miopen/perf_models/{filename}"
  if override:
    var_list = override.split(',')
    var_str = ""
    for var in var_list:
      var_str += var + " &&"

  try:
    outfile = open(os.path.expanduser(resfile), 'w+', encoding='utf-8')
    results = []
    field_names = ['Driver', 'k_time']
    writer = csv.DictWriter(outfile, fieldnames=field_names)
    writer.writeheader()
    with open(os.path.expanduser(model_path), "r", encoding='utf-8') as infile:
      for line in infile:
        try:
          idx = line.index('MIOpenDriver')
          driver_cmd = line[idx:-1]
          if override:
            cmd = f"export LD_LIBRARY_PATH={install_path}/lib && export MIOPEN_LOG_LEVEL=6 && "\
                  f"export MIOPEN_SYSTEM_DB_PATH={install_path}/share/miopen/db && "\
                  f"export MIOPEN_FIND_MODE=1 && "\
                  f"{var_str} "\
                  f"{install_path}/bin/{driver_cmd} -V 0 -i 20 -w 1 -t 1"
          else:
            cmd = f"export LD_LIBRARY_PATH={install_path}/lib && export MIOPEN_LOG_LEVEL=6 && "\
                  f"export MIOPEN_SYSTEM_DB_PATH={install_path}/share/miopen/db && "\
                  f"export MIOPEN_FIND_MODE=1 && "\
                  f"{install_path}/bin/{driver_cmd} -V 0 -i 20 -w 1 -t 1"
          print(f'Running cm: {cmd}')
          proc = subprocess.Popen(cmd,
                                  shell=True,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT)
          p_out = proc.stdout.readlines()
          k_time = -1
          for o_line in p_out:
            o_line = o_line.decode("utf-8")
            o_line = o_line.strip()
            print(o_line)
            if 'GPU Kernel Time' in o_line:
              split = re.search('Elapsed: (.*)ms', o_line)
              k_time = split.group(1)
            if 'error' in o_line:
              raise ValueError(p_out)
          results.append({'Driver': driver_cmd, 'k_time': k_time})
          print(f'k_time: {k_time}')

        except Exception as ex:
          raise ValueError(f"Could not get kernel time: {ex}")

    writer.writerows(results)
    print(f"Perf results written to: {resfile}")
    outfile.close()

  except Exception as err:
    outfile.close()
    raise ValueError(f"Could not perform performance measurement: {err}")


def compare_results(args):
  """Compare current results with previous results"""
  if not os.path.exists(results_path):
    raise ValueError(f"Results path does not exist {results_path}")
  if not os.path.exists(args.old_results_path):
    raise ValueError(f"Old results path does not exist {args.old_results_path}")

  if not compare_file(f"{results_path}/{args.filename}", \
    f"{args.old_results_path}/{args.filename}"):
    raise ValueError(f"FAILED: {args.filename}")
  print(f"PASSED: {args.filename}")


def compare_file(new_results, old_results):
  """Compare kernel_time in new vs old results file"""
  ret = True
  print(f"Comparing results from: {new_results} to {old_results}")
  with open(new_results,
            'r', encoding='utf-8') as new, open(old_results,
                                                'r',
                                                encoding='utf-8') as old:
    for line_new, line_old in zip(csv.DictReader(new), csv.DictReader(old)):
      if line_new['Driver'] != line_old['Driver']:
        print(f"New driver: {line_new['Driver']}")
        print(f"Old driver: {line_old['Driver']}")
        raise ValueError('Result files are out of sync')
      speedup = (Decimal(line_old['k_time']) - Decimal(
          line_new['k_time'])) / Decimal(line_old['k_time']) * 100
      if int(speedup) < TOLERANCE:
        print(f"{line_new['Driver']}")
        print(f"Speedup: {speedup}%")
        print(
            f"FAILED for new k_time: {line_new['k_time']} - old k_time: {line_old['k_time']}\n"
        )
        ret = False

  return ret


def main():
  """Main function"""
  args = parse_args()

  if args.compare_results:
    try:
      compare_results(args)
    except Exception as ex:
      print(f'ERR: {ex}')
      sys.exit(1)
  else:
    if not os.path.exists(results_path):
      os.makedirs(results_path)

    try:
      run_driver_cmds(f"{args.filename}", args.install_path, args.override)
    except Exception as ex:
      print(f'ERR: {ex}')
      sys.exit(1)

  return True


if __name__ == '__main__':
  main()
