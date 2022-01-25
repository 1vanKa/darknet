# -*- coding: utf-8 -*-
# @Time    : 2018/4/25 20:28
# @Author  : Adesun
# @Site    : https://github.com/Adesun
# @File    : log_parser.py

import argparse
from datetime import datetime
import logging
import os
import platform
import re
import sys
from dataclasses import dataclass
import typing as tp

# set non-interactive backend default when os is not windows
if sys.platform != 'win32':
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


@dataclass
class RegionEntry:
    normalizer_iou: float
    normalizer_obj: float
    normalizer_cls: float
    iou: float
    count: int
    class_loss: float
    iou_loss: float
    total_loss: float


@dataclass
class LogEntry:
    time: datetime
    iteration: int
    loss: float
    average_loss: float
    rate: float
    region_values: tp.List[tp.Dict[int, RegionEntry]]


def get_file_name_and_ext(filename):
    (file_path, temp_filename) = os.path.split(filename)
    (file_name, file_ext) = os.path.splitext(temp_filename)
    return file_name, file_ext


def show_message(message, stop=False):
    print(message)
    if stop:
        sys.exit(0)


def parse_args():
    parser = argparse.ArgumentParser(description="training log parser by DeepKeeper ")
    parser.add_argument('--source-dir', dest='source_dir', type=str, default='./',
                        help='the log source directory')
    parser.add_argument('--save-dir', dest='save_dir', type=str, default='./',
                        help='the directory to be saved')
    parser.add_argument('--csv-file', dest='csv_file', type=str, default="",
                        help='training log file')
    parser.add_argument('--log-file', dest='log_file', type=str, default="test_yolo.log",
                        help='training log file')
    parser.add_argument('--show', dest='show_plot', type=bool, default=True,
                        help='whether to show')
    return parser.parse_args()


def parse_between(
        text: str,
        cur_match: re.Match,
        prev_match: tp.Optional[re.Match] = None
) -> tp.List[tp.Dict[int, RegionEntry]]:
    region_pattern = re.compile(r".*: \(iou: ([\d]*\.[\d]+), obj: ([\d]*\.[\d]+), cls: ([\d]*\.[\d]+)\) "
                                r"Region ([\d]*) Avg \(IOU: ([\d]*\.[\d]+)\), count: ([\d]*), "
                                r"class_loss = ([\d]*\.[\d]+), iou_loss = ([\d]*\.[\d]+), total_loss = ([\d]*\.[\d]+)")
    if prev_match is None:
        matches = region_pattern.findall(text[:cur_match.start()])
    else:
        matches = region_pattern.findall(text[prev_match.end(): cur_match.start()])

    regions = []
    cur_regions: tp.Dict[int, RegionEntry] = {}
    last_region = -1
    for match in matches:
        cur_region = int(match[3])
        entry = RegionEntry(float(match[0]), float(match[1]), float(match[2]),
                            float(match[4]), int(match[5]), float(match[6]),
                            float(match[7]), float(match[8]))
        if cur_region < last_region:
            regions.append(cur_regions)
            cur_regions = {}
        cur_regions[cur_region] = entry
        last_region = cur_region
    return regions


def write_entry(out_file: tp.Any, entry: LogEntry) -> None:
    date_str = datetime.strftime(entry.time, "%d.%m.%YT%H:%M:%S")
    for regions in entry.region_values:
        for key, region in regions.items():
            out_file.write(f"{date_str},{entry.iteration},{entry.loss},{entry.average_loss},{entry.rate},"
                           f"{key},{region.normalizer_iou},{region.normalizer_obj},{region.normalizer_cls},"
                           f"{region.iou},{region.count},{region.iou_loss},{region.class_loss},"
                           f"{region.total_loss}\n")


def log_parser(args):
    if not args.log_file:
        show_message('log file must be specified.', True)

    log_path = os.path.join(args.source_dir, args.log_file)
    if not os.path.exists(log_path):
        show_message('log file does not exist.', True)

    file_name, _ = get_file_name_and_ext(log_path)
    log_content = open(log_path).read()

    iterations = []
    losses = []
    fig, ax = plt.subplots()
    # set area we focus on
    ax.set_ylim(0, 8)

    major_locator = MultipleLocator()
    minor_locator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(major_locator)
    ax.yaxis.set_minor_locator(minor_locator)
    ax.yaxis.grid(True, which='minor')

    pattern = re.compile(r"([\d].*): (.*?), (.*?) avg loss, (.*?) rate, .*? seconds, .*? images, .*? hours left")
    matches = pattern.finditer(log_content)
    counter = 0

    if args.csv_file != '':
        csv_path = os.path.join(args.save_dir, args.csv_file)
        out_file = open(csv_path, 'w')
    else:
        csv_path = os.path.join(args.save_dir, file_name + '.csv')
        out_file = open(csv_path, 'w')

    out_file.write("Time,Iteration,Loss,Avg Loss,Rate,Region,Norm IOU,Norm obj,Norm cls,IOU,Count,IOU Loss,Class Loss,"
                   "Total Loss\n")

    prev_match = None
    for match in matches:
        counter += 1

        if counter % 200 == 0:
            print('parsing {}'.format(counter))
        else:
            print('parsing {}'.format(counter))
        iteration, loss, avg_loss, rate = match.group(1), match.group(2), match.group(3), match.group(4)

        regions = parse_between(log_content, match, prev_match)

        iterations.append(int(iteration))
        losses.append(float(loss))
        write_entry(out_file, LogEntry(datetime.now(), int(iteration), float(loss), float(avg_loss), float(rate),
                                       regions))
        prev_match = match

    ax.plot(iterations, losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.tight_layout()

    # saved as svg
    save_path = os.path.join(args.save_dir, file_name + '.svg')
    plt.savefig(save_path, dpi=300, format="svg")
    if args.show_plot:
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    log_parser(args)
