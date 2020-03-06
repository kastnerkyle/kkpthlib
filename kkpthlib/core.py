from __future__ import print_function
import ast
import imp
import random
import numpy as np
import torch
import uuid
from scipy import linalg
from scipy.stats import truncnorm
import shutil
import socket
import os
import re
import copy
import sys
import time
import logging
from collections import OrderedDict
import hashlib
import json
import zipfile
import glob
import threading
import inspect
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
try:
    import Queue
except ImportError:
    import queue as Queue
try:
    import urllib.request as urllib  # for backwards compatibility
except ImportError:
    import urllib2 as urllib


logging.basicConfig(level=logging.INFO,
                    format='%(message)s')
logger = logging.getLogger(__name__)

string_f = StringIO()
ch = logging.StreamHandler(string_f)
# Automatically put the HTML break characters on there for html logger
formatter = logging.Formatter('%(message)s<br>')
ch.setFormatter(formatter)
logger.addHandler(ch)

def get_logger():
    return logger

sys.setrecursionlimit(40000)

# universal time
tt = str(time.time()).split(".")[0]
def get_time_string():
    return tt


def get_name():
    base = str(uuid.uuid4())
    return base


def get_script():
    py_file = None
    for argv in sys.argv[::-1]:
        if argv[-3:] == ".py":
            py_file = argv
        # slurm_script
        elif "slurm_" in argv:
            py_file = argv
    if "slurm" in py_file:
        script_name = os.environ['SLURM_JOB_NAME']
        script_name = script_name.split(".")[0]
    else:
        assert py_file is not None
        script_path = os.path.abspath(py_file)
        script_name = script_path.split(os.path.sep)[-1].split(".")[0]
        # gotta play games for slurm runner
    return script_name


# decided at import, should be consistent over training
checkpoint_uuid = get_name()[:6]
def get_checkpoint_uuid():
    return checkpoint_uuid


def set_checkpoint_uuid(uuid_str):
    logger.info("Setting global uuid to %s" % uuid_str)
    global checkpoint_uuid
    checkpoint_uuid = uuid_str


checkpoint_import_time = time.strftime("%H-%M-%S_%Y-%d-%m", time.gmtime())
def get_checkpoint_import_time():
    return checkpoint_import_time


def set_checkpoint_import_time(time_str):
    logger.info("Setting global dagbldr import time to %s" % time_str)
    global checkpoint_import_time
    checkpoint_import_time = time_str


def _special_check(verbose=True):
    ip_addr = socket.gethostbyname(socket.gethostname())
    subnet = ".".join(ip_addr.split(".")[:-1])
    whitelist = ["132.204.24", "132.204.25", "132.204.26", "132.204.27", "172.16.2"]
    subnet_match = [subnet == w for w in whitelist]
    hostname = socket.gethostname()
    if hostname == "mila00":
        # edge case for mila00
        subnet_match = [True]
    if any(subnet_match):
        if verbose:
            logger.info("Found special Mila runtime environment!")
            logger.info("IP address: %s" % ip_addr)
            logger.info("Hostname: %s" % hostname)
        return True
    else:
        return False

default_seed = 2899
logger.info("Setting all possible default seeds based on {}".format(default_seed))
# try to get deterministic runs
def seed_everything(seed=1234):
    random.seed(seed)
    tseed = random.randint(1, 1E6)
    tcseed = random.randint(1, 1E6)
    npseed = random.randint(1, 1E6)
    ospyseed = random.randint(1, 1E6)
    torch.manual_seed(tseed)
    torch.cuda.manual_seed_all(tcseed)
    np.random.seed(npseed)
    os.environ['PYTHONHASHSEED'] = str(ospyseed)
    #torch.backends.cudnn.deterministic = True

seed_everything(default_seed)

USER = os.getenv("USER")
def get_models_dir(special_check=True, verbose=True):
    checkpoint_dir = os.getenv("MODELS_DIR", os.path.join(
        os.path.expanduser("~"), "_kkpthlib_models"))

    # Figure out if this is necessary to run on localdisk @ U de M
    if special_check and _special_check(verbose=verbose):
        checkpoint_dir = "/Tmp/" + USER + "/_kkpthlib_models"
    return checkpoint_dir


def get_cache_dir(special_check=True):
    if special_check and _special_check(verbose=verbose):
        local_cache_dir = "/Tmp/" + USER + "/_kkpthlib_data_cache/"
    else:
        local_cache_dir = "/home/" + USER + "_kkpthlib_data_cache"
    if not os.path.exists(local_cache_dir):
        os.mkdir(local_cache_dir)
    return local_cache_dir


def get_lookup_dir():
    lookup_dir = os.getenv("LOOKUP_DIR", os.path.join(
        os.path.expanduser("~"), "_kkpthlib_lookup"))
    if not os.path.exists(lookup_dir):
        logger.info("LOOKUP_DIR directory {} not found, creating".format(lookup_dir))
        os.mkdir(lookup_dir)
    return lookup_dir


def _hash_file(fpath):
    assert os.path.exists(fpath)

    def md5(fname):
        hash_md5 = hashlib.md5()
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    return str(md5(fpath))


def write_lookup_file(script_path=None):
    gcu = get_checkpoint_uuid()
    gcit = get_checkpoint_import_time()
    hostname = socket.gethostname()
    lookup_path = get_lookup_dir()
    if script_path is None:
        script_name = get_script()
        full_script_path = os.path.abspath(script_name) + ".py"
    else:
        # this edge case only for making new lookups. Not recommended
        script_name = script_path.split(os.sep)[-1][:-3]
        full_script_path = script_path

    hsh = _hash_file(full_script_path)

    info_dict = {}
    info_dict["name"] = script_name
    info_dict["run_path"] = full_script_path
    info_dict["hostname"] = hostname
    info_dict["uuid"] = gcu
    info_dict["import_time"] = gcit
    info_dict["script_hash"] = hsh
    # force git commit and store that instead of all this other stuff?

    save_path = os.path.join(lookup_path, "%s_%s.json" % (gcu, script_name))
    logger.info("Saving lookup in %s" % save_path)
    with open(save_path, "w") as f:
        json.dump(info_dict, f)


def get_checkpoint_dir(checkpoint_dir=None, folder=None, create_dir=True):
    """ Get checkpoint directory path """
    if checkpoint_dir is None:
        checkpoint_dir = get_models_dir()

    if folder is None:
        checkpoint_name = get_script()
        checkpoint_import_time = get_checkpoint_import_time()
        checkpoint_uuid = get_checkpoint_uuid()
        tmp = checkpoint_dir + os.path.sep + checkpoint_name + "_" + checkpoint_import_time  + "_" + checkpoint_uuid
        checkpoint_dir = tmp
    else:
        checkpoint_dir = os.path.join(checkpoint_dir, folder)

    if not os.path.exists(checkpoint_dir) and create_dir:
        os.makedirs(checkpoint_dir)
    return checkpoint_dir


def get_resource_dir(name):
    """ Get dataset directory path """
    # Only used for JS downloader
    resource_dir = get_models_dir(verbose=False)
    resource_dir = os.path.join(resource_dir, name)
    if not os.path.exists(resource_dir):
        os.makedirs(resource_dir)
    return resource_dir


def zip_dir(src, dst):
    print("zip_dir not yet usable")
    raise ValueError()
    zf = zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED)
    abs_src = os.path.abspath(src)
    exclude_exts = [".js", ".pyc", ".html", ".txt", ".csv", ".gz"]
    for root, dirs, files in os.walk(src):
        for fname in files:
            if all([e not in fname for e in exclude_exts]):
                absname = os.path.abspath(os.path.join(root, fname))
                arcname = "tfbldr" + os.sep + absname[len(abs_src) + 1:]
                zf.write(absname, arcname)
    zf.close()


def archive_code():
    checkpoint_dir = get_checkpoint_dir()

    save_script_path = checkpoint_dir + os.path.sep + get_script() + ".py"
    script_name = get_script() + ".py"
    script_location = os.path.abspath(script_name)
    existing_reports = glob.glob(os.path.join(checkpoint_dir, "*.html"))
    empty = len(existing_reports) == 0

    if not os.path.exists(save_script_path) or empty:
        logger.info("Saving script file to {}".format(checkpoint_dir))
        shutil.copy2(script_location, save_script_path)


def coroutine(func):
    def start(*args,**kwargs):
        cr = func(*args,**kwargs)
        cr.next()
        return cr
    return start


def download(url, server_fname, local_fname=None, progress_update_percentage=5,
             bypass_certificate_check=False):
    """
    An internet download utility modified from
    http://stackoverflow.com/questions/22676/
    how-do-i-download-a-file-over-http-using-python/22776#22776
    """
    if bypass_certificate_check:
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        u = urllib.urlopen(url, context=ctx)
    else:
        u = urllib.urlopen(url)
    if local_fname is None:
        local_fname = server_fname
    full_path = local_fname
    meta = u.info()
    with open(full_path, 'wb') as f:
        try:
            file_size = int(meta.get("Content-Length"))
        except TypeError:
            logger.info("WARNING: Cannot get file size, displaying bytes instead!")
            file_size = 100
        logger.info("Downloading: %s Bytes: %s" % (server_fname, file_size))
        file_size_dl = 0
        block_sz = int(1E7)
        p = 0
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)
            if (file_size_dl * 100. / file_size) > p:
                status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl *
                                               100. / file_size)
                logger.info(status)
                p += progress_update_percentage


def filled_js_template_from_results_dict(results_dict, default_show="all"):
    # Uses arbiter strings in the template to split the template and stick
    # values in
    partial_path = get_resource_dir("js_plot_dependencies")
    full_path = os.path.join(partial_path, "master.zip")
    url = "https://github.com/kastnerkyle/simple_template_plotter/archive/master.zip"
    if not os.path.exists(full_path):
        logger.info("Downloading plotter template code from %s" % url)
        if _special_check:
            download(url, full_path, bypass_certificate_check=True)
        else:
            download(url, full_path)
        zip_ref = zipfile.ZipFile(full_path, 'r')
        zip_ref.extractall(partial_path)
        zip_ref.close()

    js_path = os.path.join(partial_path, "simple_template_plotter-master")
    template_path =  os.path.join(js_path, "template.html")
    f = open(template_path, mode='r')
    all_template_lines = f.readlines()
    f.close()
    imports_split_index = [n for n, l in enumerate(all_template_lines)
                           if "IMPORTS_SPLIT" in l][0]
    data_split_index = [n for n, l in enumerate(all_template_lines)
                        if "DATA_SPLIT" in l][0]
    log_split_index = [n for n, l in enumerate(all_template_lines)
                       if "LOGGING_SPLIT" in l][0]
    first_part = all_template_lines[:imports_split_index]
    imports_part = []
    js_files_path = os.path.join(js_path, "js")
    js_file_names = ["jquery-1.9.1.js", "knockout-3.0.0.js",
                     "highcharts.js", "exporting.js"]
    js_files = [os.path.join(js_files_path, jsf) for jsf in js_file_names]
    for js_file in js_files:
        with open(js_file, "r") as f:
            imports_part.extend(
                ["<script>\n"] + f.readlines() + ["</script>\n"])
    post_imports_part = all_template_lines[
        imports_split_index + 1:data_split_index]
    log_part = all_template_lines[data_split_index + 1:log_split_index]
    last_part = all_template_lines[log_split_index + 1:]

    def gen_js_field_for_key_value(key, values, show=True):
        assert type(values) is list
        if isinstance(values[0], (np.generic, np.ndarray)):
            values = [float(v.ravel()) for v in values]
        maxlen = 1500
        if len(values) > maxlen:
            values = list(np.interp(np.linspace(0, len(values), maxlen),
                          np.arange(len(values)), values))
        show_key = "true" if show else "false"
        return "{\n    name: '%s',\n    data: %s,\n    visible: %s\n},\n" % (
            str(key), str(values), show_key)

    data_part = [gen_js_field_for_key_value(k, results_dict[k], True)
                 if k in default_show or default_show == "all"
                 else gen_js_field_for_key_value(k, results_dict[k], False)
                 for k in sorted(results_dict.keys())]
    all_filled_lines = first_part + imports_part + post_imports_part
    all_filled_lines = all_filled_lines + data_part + log_part
    # add logging output
    tmp = copy.copy(string_f)
    tmp.seek(0)
    log_output = tmp.readlines()
    del tmp
    all_filled_lines = all_filled_lines + log_output + last_part
    return all_filled_lines


def save_results_as_html(save_path, results_dict, use_checkpoint_dir=True,
                         default_no_show="_auto", latest_tag=None):
    show_keys = [k for k in results_dict.keys()
                 if default_no_show not in k]
    as_html = filled_js_template_from_results_dict(
        results_dict, default_show=show_keys)
    if use_checkpoint_dir:
        save_path = os.path.join(get_checkpoint_dir(), save_path)
    logger.info("Saving HTML results %s" % save_path)
    with open(save_path, "w") as f:
        f.writelines(as_html)
    if latest_tag is not None:
        latest_path = os.path.join(get_checkpoint_dir(), latest_tag + "_latest.html")
        if os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(save_path, latest_path)
    logger.info("Completed HTML results saving %s" % save_path)


@coroutine
def threaded_html_writer(interp=True, maxsize=25):
    """
    Expects to be sent a tuple of (save_path, results_dict)
    """
    messages = Queue.PriorityQueue(maxsize=maxsize)
    def run_thread():
        while True:
            p, item = messages.get()
            if item is GeneratorExit:
                return
            else:
                save_path, results_dict = item
                save_results_as_html(save_path, results_dict)
    threading.Thread(target=run_thread).start()
    try:
        n = 0
        while True:
            item = (yield)
            messages.put((n, item))
            n -= 1
    except GeneratorExit:
        messages.put((1, GeneratorExit))


def save_checkpoint(state, filename):
    torch.save(state, filename)


def save_model_skeleton(serialization_dict, filename):
    model = serialization_dict["model"]
    optimizer = serialization_dict["optimizer"]
    model_source_skeleton_lines = inspect.getsourcelines(type(model))
    model_file = inspect.getsourcefile(type(model))
    with open(filename, "w") as f:
        f.write(repr(serialization_dict["hparams"]))
        f.writelines(["\n", "model from line {} of {}\n".format(model_source_skeleton_lines[1], model_file), "\n"])
        f.writelines(model_source_skeleton_lines[0])
        f.writelines(["\n", "\n"])
        f.writelines(["\n", "pytorch model representation:\n", "\n"])
        f.write(repr(model))
        f.writelines(["\n", "pytorch optimizer representation:\n", "\n"])
        f.write(repr(optimizer))


class Saver(object):
    def __init__(self, max_to_keep=5):
        self.max_to_keep = max_to_keep
        self.counter = 0

    def save(self, serialization_dict, path_stub, global_step=None):
        assert "model" in serialization_dict
        assert "hparams" in serialization_dict
        assert "optimizer" in serialization_dict
        if sorted(serialization_dict.keys()) != ["hparams", "model", "optimizer"]:
            logger.info("Detected more than just 'hparams', 'model', 'optimizer' keys in serialization dict to Saver - not currently saving anything but the model, {}".format(sorted(serialization_dict.keys())))

        if global_step is not None:
            full_model_path = path_stub + "model-{}.pth".format(global_step)
            full_optimizer_path = path_stub + "optimizer-{}.pth".format(global_step)
        else:
            full_model_path = path_stub + "model-{}.pth".format(self.counter)
            full_optimizer_path = path_stub + "optimizer-{}.pth".format(self.counter)
            self.counter += 1

        folder = "/".join(path_stub.split("/")[:-1])
        if not os.path.exists(folder):
            logger.info("Folder {} not found, creating".format(folder))
            os.makedirs(folder)
        all_files = os.listdir(folder)
        all_files = [folder + "/" + a for a in all_files]
        match_files = [a for a in all_files if path_stub in a]
        match_files = sorted(match_files, key=lambda x:int(x.split("-")[-1].split(".")[0]))
        if len(match_files) > self.max_to_keep:
            # delete oldest file, assumed sort in descending order so [0] is the oldest model
            os.remove(match_files[0])

        skeleton_path = folder + "/_model_source_skeleton.txt"
        save_model_skeleton(serialization_dict, skeleton_path)

        model = serialization_dict["model"]
        optimizer = serialization_dict["optimizer"]

        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()

        save_checkpoint(model_state_dict, full_model_path)
        save_checkpoint(optimizer_state_dict, full_optimizer_path)


# TODO: Time based saver?
def run_loop(train_loop_function, train_itr,
             valid_loop_function, valid_itr,
             serialization_dict,
             n_steps=np.inf,
             n_epochs=np.inf,
             n_train_steps_per=-1,
             train_stateful_args=None,
             n_valid_steps_per=-1,
             valid_stateful_args=None,
             status_every_s=5,
             models_to_keep=5,
             permanent_models_to_keep=100,
             permanent_step_upper_lim=50000,
             permanent_step_lower_lim=1000):
    """
    loop function signature

    r = train_loop_function(train_itr, extras, this_train_stateful_args)

    train_itr is the train iterator, used for going through dataset
    extras is a dictionary containing info like whether the model is in train or test model, is mutable / editable
    train_stateful_args are stateful arguments, which come from previous iterations of the train_loop function

    returned r should be something like

    return loss, None, stateful_args

    or

    return loss, None (if train_stateful_args or valid_stateful_args doesn't exist, you can just drop it)

    or

    return [loss1, ..., lossN], None, stateful_args

    or

    return [loss1, ..., lossN], None

    where middle argument is a "summary variable", which is either None or a dict of {"name": name_value} you wish to save in the plotter
    to force lines to plot by default, make sure the last part of the name is "_auto" such as "kl_divergence_auto"

    stateful_args is a list of variables which will be passed into the next loop iteration (useful for RNNs)

    multiple loss support is available through passing a list of losses, names will simply be values


    validation function behaves similarly, but validation values are interpolated in the plot to make arrangement smoother


    n_epochs and n_steps are exclusive! use one or the other


    serialization_dict is a dict containing (at least) {"model": pytorch_model_instance,
                                                        "optimizer": pytorch_optimizer_instance,
                                                        "hparams": kkpthlib_hparams_instance}
    TODO: support custom serialization functions
    """

    if not np.isinf(n_steps):
        if not np.isinf(n_epochs):
            raise ValueError("Both n_steps and n_epochs set - only 1 can be used! Set either n_steps or n_epochs (or both) to np.inf")

    assert "model" in serialization_dict
    assert "hparams" in serialization_dict
    assert "optimizer" in serialization_dict
    # This could be configurable, but I prefer a hard required file and name for serialization
    script = get_script()
    full_script_path = os.path.abspath(script + ".py")
    folder = str(os.sep).join(full_script_path.split(os.sep)[:-1])

    write_lookup_file()
    archive_code()

    hostname = socket.gethostname()
    logger.info("Host %s, script %s" % (hostname, script + ".py"))
    train_itr_steps_taken = 0
    valid_itr_steps_taken = 0
    overall_train_loss = []
    overall_valid_loss = []
    overall_train_summaries = {}
    overall_valid_summaries = {}
    # won't match exactly due to this - even after replaying itr stateful args may change
    # however, should be *close* since data is at least iterated in the same way...
    #this_train_stateful_args = copy.deepcopy(train_stateful_args)
    #this_valid_stateful_args = copy.deepcopy(valid_stateful_args)
    if train_stateful_args is not None:
        this_train_stateful_args = [tsa.clone() for tsa in train_stateful_args]
    else:
        this_train_stateful_args = train_stateful_args
    if valid_stateful_args is not None:
        this_valid_stateful_args = [vsa.clone() for vsa in valid_stateful_args]
    else:
        this_valid_stateful_args = valid_stateful_args
    last_status = time.time()

    model_saver = Saver(max_to_keep=models_to_keep)
    train_best_model_saver = Saver(max_to_keep=models_to_keep)
    valid_best_model_saver = Saver(max_to_keep=models_to_keep)
    perma_saver = Saver(max_to_keep=permanent_models_to_keep)

    checkpoint_dir = get_checkpoint_dir()
    thw = threaded_html_writer()

    cumulative_train_time = []
    minibatch_train_time = []
    minibatch_train_count = []
    cumulative_valid_time = []
    minibatch_valid_time = []
    minibatch_valid_count = []
    min_last_train_loss = np.inf
    min_valid_loss = np.inf
    was_best_valid_loss = False
    # todo: allow -1 to use default iterator stop point?
    if n_train_steps_per == -1:
        # this should natively use the "StopIteration" protocol of the underlying iterator
        n_train_steps_per = 1000000000
        print_n_train_steps_per = -1
    else:
        print_n_train_steps_per = n_train_steps_per

    if n_valid_steps_per == -1:
        # this should natively use the "StopIteration" protocol of the underlying iterator
        n_valid_steps_per = 1000000000
        print_n_valid_steps_per = -1
    else:
        print_n_valid_steps_per = n_valid_steps_per

    total_epochs = 0
    while True:
        if total_epochs + 1 >= n_epochs:
            break
        total_epochs += 1

        # stop at the start of an epoch
        if train_itr_steps_taken + 1 >= n_steps:
            break
        extras = {}
        extras["train"] = True
        assert n_train_steps_per >= 1
        this_train_loss = []
        this_train_summaries = {}
        train_start_time = time.time()
        for tsi in range(n_train_steps_per):
            s = time.time()
            try:
                r = train_loop_function(train_itr, extras, this_train_stateful_args)
            except StopIteration:
                break
            e = time.time()
            if train_stateful_args is not None:
                this_train_stateful_args = r[-1]
            train_loss = r[0]
            # use the first loss returned to do train best checkpoint
            try:
                all_train_loss = [float(t) for t in train_loss]
            except TypeError:
                all_train_loss = [float(train_loss)]

            train_loss = all_train_loss[0]
            # should only happen for first mb of each epoch
            if len(this_train_loss) < len(all_train_loss):
                for i in range(len(all_train_loss)):
                    this_train_loss.append([])

            # should only happen for first epoch
            if len(overall_train_loss) <  len(all_train_loss):
                for i in range(len(all_train_loss)):
                    overall_train_loss.append([])

            for i in range(len(all_train_loss)):
                this_train_loss[i].append(all_train_loss[i])
            minibatch_time = e - s
            train_time_accumulator = 0 if len(cumulative_train_time) == 0 else cumulative_train_time[-1]
            cumulative_train_time.append(minibatch_time + train_time_accumulator)
            minibatch_train_time.append(minibatch_time)
            train_summary = r[1]
            if train_summary is not None:
                for k, v in train_summary.items():
                    if k not in this_train_summaries:
                        this_train_summaries[k] = []
                    this_train_summaries[k].append(v)

            train_itr_steps_taken += 1
            minibatch_train_count.append(train_itr_steps_taken)
            if (i + 1) == n_train_steps_per or (time.time() - last_status) > status_every_s:
                logger.info("[{}, script {}] train step {}/{}, overall train step {}".format(hostname, script, tsi + 1, print_n_train_steps_per, train_itr_steps_taken))
                for n, tl in enumerate(all_train_loss):
                    logger.info("train loss {} {}, overall train average {}".format(n + 1, tl, np.mean(overall_train_loss[n] + this_train_loss[n])))
                logger.info(" ")
                last_status = time.time()
        for i in range(len(this_train_loss)):
            overall_train_loss[i] += this_train_loss[i]

        if len(this_train_summaries) > 0:
            for k in this_train_summaries:
                if k not in overall_train_summaries:
                    overall_train_summaries[k] = []
                overall_train_summaries[k] += this_train_summaries[k]

        if train_loss < min_last_train_loss:
            min_last_train_loss = train_loss
            logger.info("had best train, step {}".format(train_itr_steps_taken))
            train_best_model_saver.save(serialization_dict, os.path.join(checkpoint_dir, "saved_models", "train_model"),
                                        train_itr_steps_taken)

        extras["train"] = False
        if n_valid_steps_per > 0:
            this_valid_loss = []
            this_valid_summaries = {}
            valid_start_time = time.time()
            for vsi in range(n_valid_steps_per):
                s = time.time()
                try:
                    r = valid_loop_function(valid_itr, extras, this_valid_stateful_args)
                except StopIteration:
                    break
                e = time.time()
                if valid_stateful_args is not None:
                    this_valid_stateful_args = r[-1]
                valid_loss = r[0]
                try:
                    all_valid_loss = [float(v) for v in valid_loss]
                except TypeError:
                    all_valid_loss = [float(valid_loss)]

                valid_loss = all_valid_loss[0]
                # should only happen for first mb of each epoch
                if len(this_valid_loss) < len(all_valid_loss):
                    for i in range(len(all_valid_loss)):
                        this_valid_loss.append([])

                # should only happen for first epoch
                if len(overall_valid_loss) < len(all_valid_loss):
                    for i in range(len(all_valid_loss)):
                        overall_valid_loss.append([])

                for i in range(len(all_valid_loss)):
                    this_valid_loss[i].append(all_valid_loss[i])

                if valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    was_best_valid_loss = True
                minibatch_time = e - s
                valid_time_accumulator = 0 if len(cumulative_valid_time) == 0 else cumulative_valid_time[-1]
                cumulative_valid_time.append(minibatch_time + valid_time_accumulator)
                minibatch_valid_time.append(minibatch_time)
                valid_summary = r[1]
                if valid_summary is not None:
                    for k, v in valid_summary.items():
                        # same summary variable entries for train and valid
                        assert k in this_train_summaries
                        if k not in this_valid_summaries:
                            this_valid_summaries[k] = []
                    this_valid_summaries[k].append(v)
                valid_itr_steps_taken += 1
                minibatch_valid_count.append(valid_itr_steps_taken)
                if (i + 1) == n_valid_steps_per or (time.time() - last_status) > status_every_s:
                    logger.info("[{}, script {}] valid step {}/{}, overall valid step {}".format(hostname, script, vsi + 1, print_n_valid_steps_per, valid_itr_steps_taken))
                    for n, vl in enumerate(all_valid_loss):
                        logger.info("valid loss {} {}, overall valid average {}".format(n, vl, np.mean(overall_valid_loss[n] + this_valid_loss[n])))
                    logger.info(" ")
                    last_status = time.time()
            for i in range(len(this_valid_loss)):
                valid_interpd = [vi for vi in np.interp(np.arange(len(this_train_loss[i])), np.arange(len(this_valid_loss[i])), this_valid_loss[i])]
                overall_valid_loss[i] += valid_interpd

            if len(this_valid_summaries) > 0:
                for k in this_valid_summaries:
                    if k not in overall_valid_summaries:
                        overall_valid_summaries[k] = []
                    valid_interpd = [vi for vi in np.interp(np.arange(len(this_train_summaries[k])), np.arange(len(this_valid_summaries[k])), this_valid_summaries[k])]
                    overall_valid_summaries[k] += valid_interpd

        if train_itr_steps_taken > 1E9:
            save_html_path = "model_step_{}m.html".format(train_itr_steps_taken // 1E6)
        if train_itr_steps_taken > 1E6:
            save_html_path = "model_step_{}k.html".format(train_itr_steps_taken // 1E3)
        else:
            save_html_path = "model_step_{}.html".format(train_itr_steps_taken)

        results_dict = {}
        for i in range(len(overall_train_loss)):
            results_dict["train_loss_{}".format(i)] = overall_train_loss[i]
        if len(this_train_summaries) > 0:
            for k, vl in overall_train_summaries.items():
                results_dict["train_" + k] = overall_train_summaries[k]
        results_dict["train_minibatch_time_auto"] = minibatch_train_time
        results_dict["train_cumulative_time_auto"] = cumulative_train_time
        results_dict["train_minibatch_count_auto"] = minibatch_train_count
        # shortcut "and" to avoid edge case with no validation steps
        if len(overall_valid_loss) > 0 and len(overall_valid_loss[0]) > 0:
            for i in range(len(overall_valid_loss)):
                results_dict["valid_loss_{}".format(i)] = overall_valid_loss[i]
            for k, vl in overall_valid_summaries.items():
                results_dict["valid_" + k] = overall_valid_summaries[k]
            results_dict["valid_minibatch_time_auto"] = minibatch_valid_time
            results_dict["valid_cumulative_time_auto"] = cumulative_valid_time
            results_dict["valid_minibatch_count_auto"] = minibatch_valid_count

        thw.send((save_html_path, results_dict))

        model_saver.save(serialization_dict, os.path.join(checkpoint_dir, "saved_models", "checkpoint_model"),
                         train_itr_steps_taken)

        if np.isinf(n_steps):
            # just set it to a very large number
            tmp_n_steps = 10E6
        else:
            tmp_n_steps = n_steps
        if train_itr_steps_taken % min(permanent_step_upper_lim, max(permanent_step_lower_lim, int(tmp_n_steps // permanent_models_to_keep))) == 0:
            perma_saver.save(serialization_dict, os.path.join(checkpoint_dir, "saved_models", "permanent_model"), train_itr_steps_taken)

        if was_best_valid_loss:
            logger.info("valid saver had best valid, step {}".format(train_itr_steps_taken))
            valid_best_model_saver.save(serialization_dict, os.path.join(checkpoint_dir, "saved_models", "valid_model"),
                                        train_itr_steps_taken)
            was_best_valid_loss = False

        extras["train"] = True

    logger.info("Training complete, exiting...")
