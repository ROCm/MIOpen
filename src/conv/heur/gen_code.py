#!/usr/bin/env python3

import os
import sys
import shutil
import logging
import subprocess
import re
from string import Template
import json

const_file_match = re.compile(r"ld .*-o\s+([^\0]+\s+)")
bc_file_match = re.compile(r"opt .*-o\s+([^\0]+\s+)")
BINARY_PARAM_BIN_START_SFX = 'binary_param_bin_start'
BINARY_PARAM_BIN_END_SFX = 'binary_param_bin_end'
license_txt = \
"""/*******************************************************************************
*
* MIT License
*
* Copyright (c) 2020 Advanced Micro Devices, Inc.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*
*******************************************************************************/
"""

def check_bin(name):
  if shutil.which(name) is None:
    raise(ValueError(name + ' not found, are you in the correct docker?'))

def execute(cmd, env = None):
  if env:
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env, universal_newlines=True)
  else:
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)

  for stdout_line in iter(popen.stdout.readline, ""):
    yield stdout_line
  popen.stdout.close()
  return_code = popen.wait()
  if return_code:
    raise subprocess.CalledProcessError(return_code, cmd)

def process_file(input_file, output_file):
  # launch onnx-mlir and get the output
  check_bin('onnx-mlir')
  args = ('onnx-mlir', '--EmitLib', input_file, '-o', output_file)
  my_env = os.environ.copy()
  my_env["ONNX_MLIR_VERBOSE"] = '1'

  const_filename = None
  bc_filename = None
  for line in execute(args, my_env):
    line = line.strip()
    const_filename_regex = const_file_match.search(line)
    if const_filename_regex:
      const_filename = const_filename_regex.groups()[0].strip()
      continue
    bc_filename_regex = bc_file_match.search(line)
    if bc_filename_regex:
      bc_filename = bc_filename_regex.groups()[0].strip()
      break
  os.remove(output_file + '.so')
  return const_filename, bc_filename

def extract_main_graph(bc_filename):
  check_bin('llvm-extract')
  # args = ('llvm-extract', '-keep-const-init', '-func', 'main_graph', '-glob', 'packedConst', '--recursive', bc_filename, '-o', bc_filename)
  args = ('llvm-extract', '-func', 'main_graph', '-glob', 'packedConst', '--recursive', bc_filename, '-o', bc_filename)
  for line in execute(args):
    print(line, end="")

def compile_bc(bc_filename):
  check_bin('llc')
  out_file, _ = os.path.splitext(bc_filename)
  out_file += '.o'
  args = ('llc', '-filetype=obj', '-relocation-model=pic', '-o', out_file, bc_filename)
  for line in execute(args):
    print(line, end="")
  os.remove(bc_filename)
  return out_file

def rename_syms(bin_file, syms, prefix):
  check_bin('objcopy')
  for sym in syms:
    args = ('objcopy','--redefine-sym', '{}={}_{}'.format(sym, prefix, sym if sym[0] != '_' else sym[1:]), bin_file)
    for line in execute(args):
      print(line, end="")

def gen_bin_ptrs_decls(prefixes):
    out_txt = []
    for arch in prefixes:
        for direction in ['fwd', 'bwd', 'wrw']:
            prefix = '_'.join([direction, arch])
            out_txt.append('extern char {}_binary_param_bin_start;'.format(prefix))
            out_txt.append('extern char {}_binary_param_bin_end;'.format(prefix))
            out_txt.append('extern "C" miopen::MemRef2D {}_main_graph(float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t);'.format(prefix))
    return '\n'.join(out_txt)

def gen_getEmbeddedConstPool(prefixes):
    fn_t = Template("""
extern "C" void* ${prefix}_getEmbeddedConstPool(int64_t /*_*/)
{
    auto size    = static_cast<unsigned int>(&${bin_end} - &${bin_start});
    void* buffer = malloc(size);
    if(buffer != nullptr)
       memcpy(buffer, &${bin_start}, size);
    return buffer;
}
    """)
    res = []
    for arch in prefixes:
        for direction in ['fwd', 'bwd', 'wrw']:
          prefix = '_'.join([direction, arch])
          res.append(fn_t.substitute(bin_start = '{}_{}'.format(prefix, BINARY_PARAM_BIN_START_SFX),
              bin_end = '{}_{}'.format(prefix, BINARY_PARAM_BIN_END_SFX), prefix = prefix))
    return '\n'.join(res)

def gen_GetFeatureNames(prefixes, metadata):
    fn_t = Template("""
const std::vector<std::string>& GetFeatureNames(const Handle& handle)
{
    const auto& arch = handle.GetDeviceName();

    ${body}

}
    """)
    clause_t = Template("""
    if(arch == "${arch}")
    """)
    feature_names_t = Template("""
    {
    static const std::vector<std::string> ${arch}_feature_names = {${feature_names}};
    return ${arch}_feature_names;
    }
    """)
    el = """
    else
    {
        MIOPEN_THROW(miopenStatusNotImplemented);
    }
    """
    lst_claus = []
    lst_body = []
    for arch in prefixes:
        feature_names = metadata[arch]['feature_names']
        lst = ['"{}"'.format(x) for x in feature_names]
        lst_body.append(feature_names_t.substitute(arch= arch, feature_names = ','.join(lst)))
        lst_claus.append(clause_t.substitute(arch = arch))
    body = ''
    if(len(lst_body) > 1):
        for clause, bd in zip(lst_claus[1:], lst_body[1:]):
            body += 'else ' + clause + '\n' + bd + '\n'
    body = lst_claus[0] + '\n' + lst_body[0] + body + el

    return fn_t.substitute(body= body) + '\n'

def gen_GetSolverMap(prefixes, metadata):
    fn_t = Template("""
const std::unordered_map<int, std::string>& GetSolverMap(const Handle& handle, const ProblemDescription& problem)
{
    const auto& arch = handle.GetDeviceName();
    auto fwd = problem.direction.IsForward();
    auto bwd = problem.direction.IsBackwardData();
    auto wrw = problem.direction.IsBackwardWrW();

    ${body}
}
    """)
    clause_t = Template("""
    if(arch == "${arch}" && ${direction})
    """)
    solver_map_t= Template("""
    {
    static const std::unordered_map<int, std::string> ${arch}_solver_map = {${solver_map}};
    return ${arch}_solver_map;
    }
    """)
    el = """
    else
    {
        MIOPEN_THROW(miopenStatusNotImplemented);
    }
    """
    lst_claus = []
    lst_body = []
    for arch in prefixes:
        for direction, solver_map in metadata[arch]['solver_map'].items():
            lst  = []
            for k,v in solver_map.items():
                lst.append('{{{}, "{}"}}'.format(k, v))
            solver_map = ','.join(lst)
            lst_body.append(solver_map_t.substitute(arch=arch, solver_map = solver_map))
            lst_claus.append(clause_t.substitute(arch=arch, direction=direction))

    body = ''
    if(len(lst_body) > 1):
        for clause, bd in zip(lst_claus[1:], lst_body[1:]):
            body += 'else ' + clause + '\n' + bd + '\n'
    body = lst_claus[0] + '\n' + lst_body[0] + body + el

    return fn_t.substitute(body = body) + '\n'

def gen_GetFn(prefixes, uber_metadata, fn):
    fn_t = Template("""
const std::vector<float>& Get${fn}(const Handle& handle, const ProblemDescription& problem)
{
    const auto& arch = handle.GetDeviceName();
    auto fwd = problem.direction.IsForward();
    auto bwd = problem.direction.IsBackwardData();
    auto wrw = problem.direction.IsBackwardWrW();

    ${body}

}
    """)
    vec_t = Template("""
    {
    static const std::vector<float> ${arch}_${direction}_${fn} = {${vec}};
    return ${arch}_${direction}_${fn};
    }
    """)
    el = """
    else
    {
        MIOPEN_THROW(miopenStatusNotImplemented);
    }
    """
    clause_t = Template("""if(arch == "${arch}" && ${direction}) """)
    lst_claus = []
    lst_body = []
    for arch in prefixes:
        metadata = uber_metadata[arch]
        for direction, vector in metadata[fn].items():
            vec = [str(x) for x in vector]
            sub_body = vec_t.substitute(arch = arch, direction=direction, vec=','.join(vec), fn=fn)
            clause = clause_t.substitute(arch=arch, direction=direction)
            lst_claus.append(clause)
            lst_body.append(sub_body)
    body = ''
    if(len(lst_body) > 1):
        for clause, bd in zip(lst_claus[1:], lst_body[1:]):
            body += 'else ' + clause + '\n' + bd + '\n'
    body = lst_claus[0] + '\n' + lst_body[0] + body + el
    return fn_t.substitute(body = body, fn=fn.capitalize())

def gen_CallModel(prefixes, uber_metadata):
    fn_t = Template("""
miopen::MemRef2D CallModel(const Handle& handle, const ProblemDescription& problem, miopen::Tensor2D& features)
{
    const auto& arch = handle.GetDeviceName();
    auto fwd = problem.direction.IsForward();
    auto bwd = problem.direction.IsBackwardData();
    auto wrw = problem.direction.IsBackwardWrW();

    ${body}

}
    """)
    call_t = Template("""
    {
      return ${direction}_${arch}_main_graph(features.data(), features.data(), features.offset, features.size0, features.size1, features.stride0, features.stride1);
    }
    """)
    el = """
    else
    {
        MIOPEN_THROW(miopenStatusNotImplemented);
    }
    """
    clause_t = Template("""if(arch == "${arch}" && ${direction}) """)
    lst_claus = []
    lst_body = []
    for arch in prefixes:
        metadata = uber_metadata[arch]
        for direction in ['fwd', 'bwd', 'wrw']:
            sub_body = call_t.substitute(arch = arch, direction=direction)
            clause = clause_t.substitute(arch=arch, direction=direction)
            lst_claus.append(clause)
            lst_body.append(sub_body)
    body = ''
    if(len(lst_body) > 1):
        for clause, bd in zip(lst_claus[1:], lst_body[1:]):
            body += 'else ' + clause + '\n' + bd + '\n'
    body = lst_claus[0] + '\n' + lst_body[0] + body + el
    return fn_t.substitute(body = body)

def gen_metadata_cpp(metadata):
    prefixes = metadata.keys()
    out_file = ''
    out_file += license_txt + '\n'
    header = """
#include <miopen/conv/heur/metadata.hpp>

#include <cstring>

    """
    out_file += header
    out_file += gen_bin_ptrs_decls(prefixes)
    out_file += gen_getEmbeddedConstPool(prefixes)
    out_file += "namespace miopen {" + '\n'
    out_file += gen_GetFeatureNames(prefixes, metadata)
    out_file += gen_GetSolverMap(prefixes, metadata)
    out_file += gen_GetFn(prefixes, metadata, fn='mu')
    out_file += gen_GetFn(prefixes, metadata, fn='sigma')
    out_file += gen_CallModel(prefixes, metadata)
    out_file += "} // namespace miopen" + '\n'

    return out_file

def process_onnx_file(onnx_filename, target_location, direction, arch, rem_files = False):
    const_filename, bc_filename = process_file(onnx_filename, target_location)
    extract_main_graph(bc_filename)
    graph_obj = compile_bc(bc_filename)
    prefix = '_'.join([direction, arch])
    rename_syms(const_filename, ['_binary_param_bin_end', '_binary_param_bin_start'], prefix=prefix)
    shutil.move(const_filename, target_location + '_packed_const.o')
    rename_syms(graph_obj, ['packedConst', 'main_graph', 'getEmbeddedConstPool'], prefix=prefix)
    return const_filename, graph_obj


def main():
    direction, arch = ('fwd', 'gfx906')
    prefixes = []
    const_filename, bc_filename = process_file('1-gfx906-60.onnx', '/tmp/1-gfx906-60')
    extract_main_graph(bc_filename)
    graph_obj = compile_bc(bc_filename)
    prefix = '_'.join(direction, arch)
    rename_syms(const_filename, ['_binary_param_bin_end', '_binary_param_bin_start'], prefix =prefix)
    rename_syms(graph_obj, ['packedConst', 'main_graph', 'getEmbeddedConstPool'], prefix=prefix)

if __name__ == '__main__':
    main()
