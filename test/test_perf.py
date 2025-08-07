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
import os
import re
import subprocess
import argparse
import csv
import time
from decimal import Decimal
import multiprocessing as mp

curr_path = os.path.abspath(os.getcwd())
default_results_path = curr_path + "/perf_results"
TOLERANCE = -10  #tolerance 10%

re_Elapsed = re.compile(r"(\d*\.*\d+)")
re_Solver = re.compile(r"^MIOpen .* Algorithm: (\d+), Solution: (\d+)/(\w+)")
re_GPU = re.compile(r"^GPU Kernel Time .* Elapsed: (\d+\.\d+) ms")
re_Key = re.compile(r"^.*Key match: ([\w\-]*)")

queue = mp.Queue()  #gpu idx queue, used in HIP_VISIBLE_DEVICES
results_queue = mp.Queue()


class Manager(mp.Process):
  """Queue manager"""

  def __init__(self, **kwargs):
    allowed_keys = set(['filename', 'install_path', 'override', 'results_path'])
    self.filename = None
    self.install_path = None
    self.override = False
    self.results_path = f"{default_results_path}"
    self.__dict__.update(
        (key, value) for key, value in kwargs.items() if key in allowed_keys)

    self.in_qs = {}
    self.num_gpus = int(self.get_num_gpus())
    self.resfile = f"{self.results_path}/{self.filename}"
    print(self.resfile)
    print('install_path: ', self.install_path)
    self.model_path = f"{self.install_path}/share/miopen/perf_models/{self.filename}"
    self.driver_cmds = []
    self.set_driver_cmds()
    self.writer = None

  def get_num_gpus(self):
    """Get num_gpus"""
    cmd = "/opt/rocm/bin/rocminfo | grep 'Device Type: *GPU' | wc -l"
    proc = subprocess.Popen(cmd,
                          shell=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    output = proc.communicate()[0]
    return int(output.decode('utf-8').strip())

  def set_driver_cmds(self):
    if self.override:
      var_list = self.override.split(',')
      var_str = ""
      for var in var_list:
        var_str += var + " &&"

    with open(os.path.expanduser(self.model_path), "r",
              encoding='utf-8') as infile:
      for line in infile:
        try:
          if line.find('MIOpenDriver') == -1:
            print(f"Skipping line '{line}'")
            continue
          idx = line.index('MIOpenDriver')
          driver_cmd = line[idx:-1]
          if self.override:
            cmd = f"export LD_LIBRARY_PATH={self.install_path}/lib && export MIOPEN_LOG_LEVEL=6 && "\
                  f"export MIOPEN_SYSTEM_DB_PATH={self.install_path}/share/miopen/db && "\
                  f"export HIP_VISIBLE_DEVICES=GPU_ID && "\
                  f"{var_str} "\
                  f"{self.install_path}/bin/{driver_cmd} -V 0 -i 10 -w 1 -t 1 -G 1"
          else:
            cmd = f"export LD_LIBRARY_PATH={self.install_path}/lib && export MIOPEN_LOG_LEVEL=6 && "\
                  f"export MIOPEN_SYSTEM_DB_PATH={self.install_path}/share/miopen/db && "\
                  f"export HIP_VISIBLE_DEVICES=GPU_ID && "\
                  f"{self.install_path}/bin/{driver_cmd} -V 0 -i 10 -w 1 -t 1 -G 1"
          print(f'Appending cm: {cmd}')
          self.driver_cmds.append(cmd)

        except Exception as err:
          print(f"Could not get driver commands: {err}")

    print('#driver commands: ', len(self.driver_cmds))

  def run(self):
    """Main function to launch worker pool"""
    for gpu_id in range(self.num_gpus):
      queue.put(gpu_id)
    self.launch_pool()

  def write_to_file(self, results):
    """Write results to csv file"""
    field_names = [
        'Driver', 'k_time', 'wall_time', 'solver_id', 'solver_name', 'fdb_key'
    ]
    with open(os.path.expanduser(self.resfile), 'w+', encoding='utf-8') as outfile:
      self.writer = csv.DictWriter(outfile, fieldnames=field_names)
      self.writer.writeheader()
      try:
        self.writer.writerows(results)
        print(f"Perf results written to: {self.resfile}")
      except Exception as exp:
        print(exp)

  def launch_pool(self):
    """Launch pool of driver cmds"""
    pool = mp.Pool(processes=self.num_gpus)
    for result in pool.imap_unordered(self.run_driver_cmd, self.driver_cmds):
      self.parse_result(result)
    pool.close()
    print(f"Size of results Q: {results_queue.qsize()}")
    results = []
    while not results_queue.empty():
      results.append(results_queue.get())
    results.sort(key=lambda x: x['Driver'])
    self.write_to_file(results)

  def parse_result(self, result):
    """Potential to use result as it becomes available"""
    #print(result)

  def run_driver_cmd(self, driver_cmd):
    """Launch each driver cmd in subproc"""
    while queue.empty():
      time.sleep(2)
      print('GPUs busy, sleeping')
    gpu_id = queue.get()
    cmd = driver_cmd.replace('GPU_ID', str(gpu_id))
    try:
      print("")
      print(f"Starting process on GPU {gpu_id}")
      print(cmd)

      res_dict = {}
      proc = subprocess.Popen(cmd,
                              shell=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
      p_out = proc.stdout.readlines()
      res = None
      e = Entry()
      for line in p_out:
        line = line.decode("utf-8")
        line = line.strip()
        if (line.find('MIOpenDriver') != -1) and (line.find('MIOpen(HIP)') == -1):  #fragile solution
          e.cmd = line
          print(e.cmd)
          continue
        if line.find('Wall-clock Time') != -1:
          res = re_Elapsed.findall(line)
          print(res)
          e.wall_elapsed = res[0]
          e.wall_aux = res[1]
          e.wall_gwss = res[2]
          continue
        if re_Solver.match(line):
          res = re_Solver.findall(line)[0]
          print(res)
          e.algo = res[0]
          e.sol_id = res[1]
          e.sol_name = res[2]
          continue
        if re_Key.match(line):
          e.fdb_key = re_Key.findall(line)[0]
        if re_GPU.match(line):
          res = re_GPU.findall(line)
          e.sol_time = res[0]
          print('k_time: ', e.sol_time)
        if line.find('error') != -1:
          raise ValueError(p_out)

      res_dict = {
        'Driver': e.cmd,
        'k_time': e.sol_time,
        'wall_time': e.wall_elapsed,
        'solver_id': e.sol_id,
        'solver_name': e.sol_name,
        'fdb_key': e.fdb_key
      }

      results_queue.put(res_dict)
      ret = res_dict
    finally:
      queue.put(gpu_id)
    return ret


class Entry:
  """Module to hold runtime values"""

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
  parser.add_argument('--results_path',
                      dest='results_path',
                      default=default_results_path,
                      type=str,
                      help='Specify full path to output results directory')
  parser.add_argument('--override',
                      dest='override',
                      type=str,
                      help='Specify driver cmd env vars')
  args = parser.parse_args()

  if args.compare_results and not args.old_results_path:
    parser.error(
        '--old_results_path and --compare_results must both be specified')

  return args


def compare_results(args):
  """Compare current results with previous results"""
  if not os.path.exists(args.results_path):
    raise ValueError(f"Results path does not exist {args.results_path}")
  if not os.path.exists(args.old_results_path):
    raise ValueError(f"Old results path does not exist {args.old_results_path}")

  if not compare_file(f"{args.results_path}/{args.filename}", \
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
    csv_new = list(csv.DictReader(new))
    csv_new.sort(key=lambda x: x['Driver'])
    csv_old = list(csv.DictReader(old))
    csv_old.sort(key=lambda x: x['Driver'])
    for line_new, line_old in zip(csv_new, csv_old):
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
      return False
  else:
    if not os.path.exists(args.results_path):
      os.makedirs(args.results_path)

    try:
      manager = Manager(filename=args.filename,
                        install_path=args.install_path,
                        override=args.override,
                        results_path=args.results_path)
      manager.run()
    except Exception as ex:
      print(f'ERR: {ex}')
      return False

  return True


if __name__ == '__main__':
  main()
