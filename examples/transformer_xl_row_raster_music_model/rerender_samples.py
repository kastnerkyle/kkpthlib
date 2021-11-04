import argparse
import os

from kkpthlib import music_json_to_midi

parser = argparse.ArgumentParser(description="script {}".format(__file__))
parser.add_argument('sampled_outputs_path', type=str, default="")
args = parser.parse_args()
if args.sampled_outputs_path == "":
    parser.print_help()
    sys.exit(1)

sampled_outputs_path = args.sampled_outputs_path
if not sampled_outputs_path.endswith(os.sep):
    sampled_outputs_path = sampled_outputs_path + os.sep

rerender_path = "rerenders_" + sampled_outputs_path

input_files = [sampled_outputs_path + f for f in os.listdir(sampled_outputs_path)]

if os.path.exists(rerender_path):
    raise ValueError("Folder {} already exists! Please remove before running re-render script for {}".format(rerender_path, sampled_outputs_path))

if not os.path.exists(rerender_path):
    os.mkdir(rerender_path)

input_json = [i_f for i_f in input_files if i_f.endswith(".json")]
if len(input_json) == 0:
    raise ValueError("No .json files found in {}, check that .json files exist in that directory!".format(sampled_outputs_path))

for ij in input_json:
    base_fpath = ij.split(os.sep)[-1]
    base_name = base_fpath.split(".json")[0]
    midi_name = base_name + ".midi"
    midi_out_fpath = rerender_path + midi_name

    #a = "Harpsichord"
    #b = "Harpsichord"
    #c = "Harpsichord"
    #d = "Harpsichord"
    #e = "Oboe"
    #f = "English Horn"
    #g = "Clarinet"
    #h = "Bassoon"

    # key: voice
    # values: list of tuples (instrument, time_in_quarter_notes_to_start_using) - optionally instrument, time_in_quarters, default_voice_amplitude
    #m = {0: [(a, 0), (e, 8)],
    #     1: [(b, 0), (f, 8)],
    #     2: [(c, 0), (g, 8)],
    #     3: [(d, 0), (h, 8)]}
    #m = {0: [(a, 0, 60), (e, 8, 0)],
    #     1: [(b, 0, 30), (f, 8, 0)],
    #     2: [(c, 0, 30), (g, 8, 0)],
    #     3: [(d, 0, 40), (h, 8, 50)]}
    a = "harpsichord_preset"
    #b = "woodwind_preset"
    #b = "organ_preset"
    # TODO: rewrite amplitudes so that everything non onset is a rest for piano / percussion in general?
    b = "grand_piano_preset"
    #b = "electric_piano_preset"
    #b = "zelda_preset"
    #b = "dreamy_preset"
    #b = "dreamy_r_preset"
    #b = "nylon_preset"
    m = {0: [(a, 0), (b, 8)],
         1: [(a, 0), (b, 8)],
         2: [(a, 0), (b, 8)],
         3: [(a, 0), (b, 8)]}
    music_json_to_midi(ij, midi_out_fpath,
                       voice_program_map=m)
