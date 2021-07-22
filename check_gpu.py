import os
import pexpect
import multiprocessing as mp
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_name_file", type=str)
    parser.add_argument("--password", type=str)
    return parser.parse_args()


def ssh(server_name, command, password):
    if '\n' not in password:
        password = password + '\n'
    ssh_cmd = 'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null {} {}'.format(server_name, command)
    stdout = pexpect.run(ssh_cmd, events={'(?i)password': password})
    stdout = str(stdout, "utf-8")
    stdout = stdout.replace('\\n', '\n').replace('\\r', '\r')
    return stdout


def parse_cells(cache_list):
    cells = []
    for cache in cache_list:
        cells += [it.strip() for it in cache.strip()[1:-1].split("|")]
    gpu_name = " ".join(cells[0].split()[1:-1]).strip()
    gpu_memory = [int(it.strip()[:-3]) for it in cells[4].split("/")]
    gpu_util = int(cells[5].split()[0].strip()[:-1])
    return gpu_name, gpu_memory, gpu_util


def parse_nvidia_smi(stdout):
    info = {"meta": dict(), "gpus": []}
    stdout = stdout.split("\n")
    n = 0
    parse_flag = 0
    while n < len(stdout):
        s = stdout[n]
        if s.startswith("| NVIDIA-SMI"):
            s = s.strip()[1:-1]
            items = s.split()
            info["meta"][items[0]] = items[1]
            info["meta"][items[2]+' '+items[3]] = items[4]
            info["meta"][items[5]+' '+items[6]] = items[7]
            n += 1
            parse_flag = 1
        elif parse_flag == 1 and s.startswith("|======"):
            cache_list = []
            n += 1
            while not stdout[n].startswith("+"):
                cache_list.append(stdout[n])
                n += 1
            info["gpus"].append(parse_cells(cache_list))
            parse_flag = 2
        elif parse_flag == 2 and s.startswith("+"):
            cache_lines = []
            n += 1
            while not stdout[n].startswith("+"):
                cache_list.append(stdout[n])
                n += 1
            info["gpus"].append(parse_cells(cache_list))
            if not stdout[n+1].strip():
                parse_flag = 0
        else:
            n += 1

    return info


def load_server(fname):
    server_list = []
    with open(fname) as fin:
        for line in fin:
            server_list.append(line.strip())
    return server_list


def main(args):
    server_list = load_server(args.server_name_file)
    for server in server_list:
        out = ssh(server, "nvidia-smi", args.password)
        server_gpu_info = parse_nvidia_smi(out)
        av_num = 0
        total_num = len(server_gpu_info["gpus"])
        total_util = 0
        for gpu in server_gpu_info["gpus"]:
            if gpu[1][0] == 0:
                av_num += 1
            else:
                total_util += gpu[2]

        ret = "[{}] Available/Total: {}/{}\tGPU Name: {}\tGPU Mem: {}Mib".format(
            server, av_num, total_num, server_gpu_info["gpus"][0][0], server_gpu_info["gpus"][0][1][1])
        if av_num != total_num:
            avg_util = total_util / (total_num - av_num)
            ret = "{}\tAVG. GPU-Util: {}%".format(ret, avg_util)
        print(ret)

if __name__ == "__main__":
    args = parse_args()
    main(args)
