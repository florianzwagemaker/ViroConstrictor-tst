# pylint: disable=C0103

"""
Construct and write configuration files for SnakeMake
"""

import multiprocessing
import os

import yaml


def set_cores(cores):
    available = multiprocessing.cpu_count()
    if cores == available:
        return cores - 2
    if cores > available:
        return available - 2
    return cores


def get_max_local_mem():
    avl_mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return int(round(avl_mem_bytes / (1024.0 ** 2) - 2000, -3))


def SnakemakeConfig(conf, cores, dryrun):
    cores = set_cores(cores)
    compmode = conf["COMPUTING"]["compmode"]

    if compmode == "local":
        config = {
            "cores": cores,
            "latency-wait": 60,
            "use-conda": True,
            "dryrun": dryrun,
            "jobname": "ViroConstrictor_{name}.jobid{jobid}",
        }

    if compmode == "grid":
        queuename = conf["COMPUTING"]["queuename"]
        threads = "{threads}"
        mem = "{resources.mem_mb}"
        config = {
            "cores": 300,
            "latency-wait": 60,
            "use-conda": True,
            "dryrun": dryrun,
            "jobname": "ViroConstrictor_{name}.jobid{jobid}",
            "drmaa": f' -q {queuename} -n {threads} -R "span[hosts=1]" -M {mem}',
            "drmaa-log-dir": "logs/drmaa",
        }

    return config


def SnakemakeParams(conf, cores, ref, prim, feats, platform, samplesheet, amplicon, pr_mm_rate):
    if conf["COMPUTING"]["compmode"] == "local":
        threads_highcpu = int(cores - 2)
        threads_midcpu = int(cores / 2)
        threads_lowcpu = 1
    if conf["COMPUTING"]["compmode"] == "grid":
        threads_highcpu = 12
        threads_midcpu = 6
        threads_lowcpu = 2

    params = {
        "sample_sheet": samplesheet,
        "computing_execution": conf["COMPUTING"]["compmode"],
        "max_local_mem": get_max_local_mem(),
        "reference_file": ref,
        "primer_file": prim,
        "features_file": feats,
        "platform": platform,
        "amplicon_type": amplicon,
        "primer_mismatch_rate": pr_mm_rate,
        "threads": {
            "Alignments": threads_highcpu,
            "QC": threads_midcpu,
            "AdapterRemoval": threads_lowcpu,
            "PrimerRemoval": threads_highcpu,
            "Consensus": threads_midcpu,
            "Index": threads_lowcpu,
            "Typing": threads_lowcpu,
        },
        "runparams": {
            "alignmentfilters": "-F 256 -F 512 -F 4 -F 2048",
            "qc_filter_illumina": 20,
            "qc_filter_nanopore": 7,
            "qc_filter_iontorrent": 20,
            "qc_window_illumina": 5,
            "qc_window_nanopore": 20,
            "qc_window_iontorrent": 15,
            "qc_min_readlength": 100,
        },
    }

    return params


def WriteConfigs(
    conf, cores, cwd, platform, ref, prims, feats, samplesheet, amplicon_type, pr_mm_rate, dryrun
):
    if not os.path.exists(cwd + "/config"):
        os.makedirs(cwd + "/config")

    os.chdir(cwd + "/config")

    with open("config.yaml", "w") as ConfigOut:
        yaml.dump(
            SnakemakeConfig(conf, cores, dryrun), ConfigOut, default_flow_style=False
        )
    ConfigOut.close()

    with open("params.yaml", "w") as ParamsOut:
        yaml.dump(
            SnakemakeParams(
                conf, cores, ref, prims, feats, platform, samplesheet, amplicon_type, pr_mm_rate
            ),
            ParamsOut,
            default_flow_style=False,
        )
    ParamsOut.close()

    parameters = os.getcwd() + "/params.yaml"
    snakeconfig = os.getcwd() + "/config.yaml"
    return parameters, snakeconfig


def LoadConf(configfile):
    with open(configfile, "r") as ConfIn:
        conf = yaml.load(ConfIn, Loader=yaml.FullLoader)
    return conf
