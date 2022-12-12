#!/usr/bin/env python
from pathlib import Path


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
            if line_info[0][-1] != ':':
                print(line_info[0])
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
    return result


def main():
    parse_file('tests_1060.txt')


if __name__ == '__main__':
    main()
