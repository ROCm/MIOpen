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

re_Elapsed = re.compile(r"(\d*\.*\d+)")
re_Solver = re.compile(r"^MIOpen .* Algorithm: (\d+), Solution: (\d+)/(\w+)")
re_GPU = re.compile(r"^GPU Kernel Time .* Elapsed: (\d+\.\d+) ms")
re_Key = re.compile(r"^.*Key match: ([\w\-]*)")


class Entry:

  def __init__(self):
    self.cmd = ''
    self.wall_elapsed = ''
    self.wall_aux = ''
    self.wall_gwss = ''
    self.algo = ''
    self.sol_id = ''
    self.sol_name = ''
    self.sol_time = ''
    self.fdb_key = ''

  def __str__(self):
    atrs = [
        self.cmd, self.wall_elapsed, self.wall_aux, self.wall_gwss, self.algo,
        self.sol_id, self.sol_name, self.sol_time, self.fdb_key
    ]
    return ",".join(atrs)


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
    field_names = [
        'Driver', 'k_time', 'wall_time', 'solver_id', 'solver_name', 'fdb_key'
    ]
    writer = csv.DictWriter(outfile, fieldnames=field_names)
    writer.writeheader()
    with open(os.path.expanduser(model_path), "r", encoding='utf-8') as infile:
      for line in infile:
        try:
          if (line.find('MIOpenDriver') == -1):
            print(f"Skipping line '{line}'")
            continue
          idx = line.index('MIOpenDriver')
          driver_cmd = line[idx:-1]
          if override:
            cmd = f"export LD_LIBRARY_PATH={install_path}/lib && export MIOPEN_LOG_LEVEL=6 && "\
                  f"export MIOPEN_SYSTEM_DB_PATH={install_path}/share/miopen/db && "\
                  f"{var_str} "\
                  f"{install_path}/bin/{driver_cmd} -V 0 -i 10 -w 1 -t 1"
          else:
            cmd = f"export LD_LIBRARY_PATH={install_path}/lib && export MIOPEN_LOG_LEVEL=6 && "\
                  f"export MIOPEN_SYSTEM_DB_PATH={install_path}/share/miopen/db && "\
                  f"{install_path}/bin/{driver_cmd} -V 0 -i 10 -w 1 -t 1"
          print(f'Running cm: {cmd}')
          proc = subprocess.Popen(cmd,
                                  shell=True,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT)
          p_out = proc.stdout.readlines()
          k_time = -1
          res = None
          e = None
          for line in p_out:
            line = line.decode("utf-8")
            line = line.strip()
            print(line)
            if (line.find('MIOpenDriver') != -1):
              e = Entry()
              e.cmd = line
              continue
            if (line.find('Wall-clock Time') != -1):
              res = re_Elapsed.findall(line)
              e.wall_elapsed = res[0]
              e.wall_aux = res[1]
              e.wall_gwss = res[2]
              continue
            if (re_Solver.match(line)):
              res = re_Solver.findall(line)[0]
              e.algo = res[0]
              e.sol_id = res[1]
              e.sol_name = res[2]
              continue
            if (re_Key.match(line)):
              e.fdb_key = re_Key.findall(line)[0]
            if (re_GPU.match(line)):
              res = re_GPU.findall(line)
              e.sol_time = res[0]
              print(e)
            if line.find('error') != -1:
              raise ValueError(p_out)
          results.append({
              'Driver': e.cmd,
              'k_time': e.sol_time,
              'wall_time': e.wall_elapsed,
              'solver_id': e.sol_id,
              'solver_name': e.sol_name,
              'fdb_key': e.fdb_key
          })
          print(f'k_time: {e.sol_time}')

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
      if line_new['solver_name'] != line_old['solver_name']:
        print(
            f"Winning solver changed from {line_new['solver_name']} to {line_old['solver_name']}"
        )
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
