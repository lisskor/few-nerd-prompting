import argparse
import json

from typing import List


def process_files(input_files: List[str], output_file: str) -> None:
    """
    Process a list of input files containing JSON-formatted lines,
    extract numeric keys from each JSON object, sort lines based on
    the keys, write the sorted lines to the output file.

    :param input_files: list of input files
    :param output_file: path to write merged and sorted output
    :return: None
    """
    all_lines = []

    # Read lines from each input file
    for input_file in input_files:
        with open(input_file, 'r', encoding='utf8') as file:
            lines = file.readlines()
            all_lines.extend(lines)

    # Parse JSON and extract numeric keys
    parsed_lines = []
    for line in all_lines:
        try:
            data = json.loads(line)
            key = int(list(data.keys())[0])
            parsed_lines.append((key, line))
        except (json.JSONDecodeError, IndexError, ValueError):
            print(f"Skipping invalid JSON line: {line}")

    # Sort lines based on numeric keys
    sorted_lines = sorted(parsed_lines, key=lambda x: x[0])

    # Write sorted lines to the output file
    with open(output_file, 'w') as out_file:
        for _, line in sorted_lines:
            out_file.write(line)


if __name__ == '__main__':
    # Add arguments to argparser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input_files',
        type=str,
        nargs='+',
        help='Input result files.'
    )
    parser.add_argument(
        '-o', '--output_file',
        default='results.out',
        type=str,
        help='Output file.'
    )

    arguments = parser.parse_args()
    process_files(arguments.input_files, arguments.output_file)
