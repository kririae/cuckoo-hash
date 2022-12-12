#!/usr/bin/env python
from pathlib import Path
import statistics as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib as mpl


def parse_file(filename: Path):
    filename = Path(filename)
    lines = None
    with filename.open() as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]

    # state machine
    results = []
    result = {}
    prog = False
    test_name = None
    for line in lines:
        line_info = line.split(' ')
        if line.startswith('>>>'):
            prog = True
            test_name = line_info[1]
            result['test_name'] = test_name
        elif line.startswith('<<<'):
            # commit current result
            prog = False
            test_name = None
            results.append(result)
            result = {}
        elif prog:
            assert line_info[0][-1] == ':'
            param_name = ''.join(list(line_info[0])[:-1])
            param_value = line_info[1]
            if param_value == 'false':
                param_value = False
            elif param_value == 'true':
                param_value = True
            elif '.' in param_value:
                param_value = float(param_value)
            else:
                param_value = int(param_value)
            result[param_name] = param_value
    assert prog == False
    return results


def graph_test_1(results, n: int, gpu_name: str):
    results = [i for i in results if i.get('test_name') == 'test_insertion']
    results_hash_functions = [
        i for i in results if i.get('num_hash_functions') == n]
    s = list(range(10, 25))
    MOPs = []
    for sp in s:
        # collect corresponding values
        result_sp = [i.get('build_MOPs')
                     for i in results_hash_functions if i.get('num_keys') == (1 << sp)]
        MOPs.append(st.mean(result_sp))
    plt.plot(s, MOPs, '.-', label=f'$n={n}$ on {gpu_name}')
    plt.grid(True)
    plt.title(f'Parallel HashTable Build MOPs on GPUs')
    plt.xlabel('$s$')
    plt.ylabel('MOPs (million operations)')
    plt.legend()


def graph_test_2(results, n: int, gpu_name: str):
    results = [i for i in results if i.get('test_name') == 'test_lookup']
    results_hash_functions = [
        i for i in results if i.get('num_hash_functions') == n]
    exist_proportion = list(np.linspace(0.1, 1, num=10))
    MOPs = []
    for e in exist_proportion:
        result_e = [i.get('query_MOPs') for i in results_hash_functions if abs(i.get('exist_proportion') - e) < 1e-4]
        MOPs.append(st.mean(result_e))
    plt.plot(exist_proportion, MOPs, '.-', label=f'$n={n}$ on {gpu_name}')
    plt.grid(True)
    plt.title(f'HashTable Query MOPs on GPUs')
    plt.xlabel('$i$')
    plt.ylabel('MOPs (million operations)')
    plt.legend()
    


def main():
    tests = [('tests_3090.txt', 'RTX 3090 24G'), ('tests_3060.txt', 'RTX 3060 8G'),
             ('tests_1660ti.txt', 'GTX 1660 Ti 6G'), ('tests_1060.txt', 'GTX 1060 6G')]
    # test 1
    for filename, gpu_name in tests:
        results = parse_file(filename)
        graph_test_1(results, 3, gpu_name)
    for filename, gpu_name in tests:
        results = parse_file(filename)
        graph_test_1(results, 2, gpu_name)
    plt.tight_layout()
    plt.savefig('report/fig/test_1.pdf')
    plt.close()

    # test 2
    for filename, gpu_name in tests:
        results = parse_file(filename)
        graph_test_2(results, 3, gpu_name)
        break
    plt.savefig('report/fig/test_2_3090.pdf')
    plt.close()
    for filename, gpu_name in tests:
        results = parse_file(filename)
        graph_test_2(results, 3, gpu_name)
    plt.savefig('report/fig/test_2_all.pdf')
    plt.close()


if __name__ == '__main__':
    font_dir = ['/usr/share/fonts/OTF/']
    for font in font_manager.findSystemFonts(font_dir):
        print(font)
        font_manager.fontManager.addfont(font)
    mpl.rcParams['font.family'] = 'Latin Modern Roman'
    mpl.rcParams['font.size'] = '16'
    mpl.rcParams['figure.figsize'] = 10, 10
    main()
