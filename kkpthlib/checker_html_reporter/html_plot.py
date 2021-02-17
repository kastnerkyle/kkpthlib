# Author: Kyle Kastner
# webpage parts based heavily on examples from indirajhenny
# http://bl.ocks.org/indirajhenny/15d7218cadf4fa96e407f87f034b1ff7
import os

pre = '''
var notes = [
'''

pre_chunk = '''{
'''

chunk = '''    "name": "{}",
    "midi": {},
    "time": {},
    "velocity": {},
    "duration": {}
'''

post_chunk = '''},
'''

post = '''
];
'''

# need to auto-generate this...
code_stub_pre = '''
var midiOnly = [];

notes.forEach(function(item) {
    //get value of name
    var val = item.midi;
    //push duration value into array
    midiOnly.push(val);
});

//console.log(midiOnly);
'''

# create empty lists, as many as we have note range (start with 21?)
# create notes function and final function

N_LANES = 108 - 20

midi_to_name_lookup = {}
note_bases = {0: "C",
              1: "C#",
              2: "D",
              3: "Eb",
              4: "E",
              5: "F",
              6: "F#",
              7: "G",
              8: "Ab",
              9: "A",
              10: "Bb",
              11: "B"}

for i in range(20, 108):
    b = note_bases[i % 12]
    o = (i // 12) - 1
    midi_to_name_lookup[i] = b + str(o)

"""
LANES_LOOKUP= {20: "C3",
               19: "D3",
               18: "E3",
               17: "F3",
               16: "G3",
               15: "A3",
               14: "B3",

               13: "C4",
               12: "D4",
               11: "E4",
               10: "F4",
               9: "G4",
               8: "A4",
               7: "B4",

               6: "C5",
               5: "D5",
               4: "E5",
               3: "F5",
               2: "G5",
               1: "A5",
               0: "B5",
              }
"""
LANES_LOOKUP = {}
for _n, i in enumerate(range(20, 108)):
    rev = list(range(20, 108))[::-1]
    LANES_LOOKUP[i - 20] = midi_to_name_lookup[rev[_n]]

def _create_code_stub_init():
    s = "\n"
    template = "minimalNotes{} = [];\n"
    for i in range(N_LANES):
        s = s + template.format(i)
    s = s + "minimalNotesFinal = [];\n"
    return s
code_stub_init = _create_code_stub_init()

def _create_code_stub_functions():
    # use simple subs because the string has both {} and ()
    from string import Template
    s = "\n"
    template = '''
minimalNotes$index = notes.filter(function (el){
    return el.name == "$note_name";
});

var minimalNotesFinal$index = minimalNotes$index.map(function(el){
    var addKey = Object.assign({}, el);
    addKey.lane = $lane_index;
    return addKey;
});
'''
    t = Template(template)
    for i in range(N_LANES):
        s = s + t.substitute(index=i, lane_index=i, note_name=LANES_LOOKUP[i])
    return s

code_stub_functions = _create_code_stub_functions()

def _create_code_stub_combine():
    s = "\n"
    template = "minimalNotesFinal = minimalNotesFinal.concat(minimalNotesFinal{})\n"
    for i in range(N_LANES):
        s = s + template.format(i)
    s + "console.log(minimalNotesFinal)\n"
    return s

code_stub_combine = _create_code_stub_combine()

def make_chunk(note_tuple):
    midi = note_tuple[0]
    name = midi_to_name_lookup[midi]
    start_time = note_tuple[1]
    velocity = 1.0
    duration = note_tuple[2]
    s = pre_chunk + chunk.format(name, midi, start_time, velocity, duration) + post_chunk
    return s

code_stub = code_stub_pre + code_stub_init + code_stub_functions + code_stub_combine
def make_plot_json(list_of_notes):
    cur = pre
    # voice track?
    for note in list_of_notes:
        c = make_chunk(note)
        cur = cur + c
    return cur[:-2] + post + code_stub

def make_website_string(javascript_note_data_string, page_name="Piano Roll Plot", end_time=60):
    from string import Template
    with open("report_template.html", "r") as f:
        l = f.read()
    t = Template(l)
    # if we reverse the list, we reverse the axis
    return t.substitute(PAGE_NAME=page_name, JAVASCRIPT_NOTE_DATA=javascript_note_data_string, LANE_NAMES=str([LANES_LOOKUP[i] for i in range(N_LANES)]), LANE_TIME_END=end_time)


def make_index_html_string(list_of_report_file_base_names):
    from string import Template
    with open("index_template.html", "r") as f:
        l = f.read()

    report_div_template = '<div id="{}"></div>\n'
    report_load_template = "    $('#{}').load('{}.html');\n"

    index_chunk = "\n"
    # do divs
    for name in list_of_report_file_base_names:
        index_chunk = index_chunk + report_div_template.format(name)
    # do script tag
    index_chunk = index_chunk + "<script>\n"
    # do loads
    for name in list_of_report_file_base_names:
        index_chunk = index_chunk + report_load_template.format(name, name)
    # end script
    index_chunk = index_chunk + "</script>\n"

    # unused example of what it should look like
    """
    example_index_chunk = '''
    <div id="report1"></div>
    <div id="report2"></div>

    <script>
        $('#report1').load('report1.html');
        $('#report2').load('report2.html');
    </script>
    '''
    """
    t = Template(l)
    return t.substitute(INDEX_BODY=index_chunk)


def test_example():
    qppm = 220
    #qpps = qppm / 60.
    quarter_time_const_s = .5
    # notes should be note value, start time, duration
    # we add the name internally to be consistent
    import json
    with open("bwv101.7.C-minor-transposed.json", "r") as f:
        music_json_data = json.load(f)

    l = []
    for _p in range(len(music_json_data["parts"])):
        parts = music_json_data["parts"][_p]
        parts_times = music_json_data["parts_times"][_p]
        parts_cumulative_times = music_json_data["parts_cumulative_times"][_p]
        assert len(parts) == len(parts_times)
        assert len(parts_times) == len(parts_cumulative_times)
        for _s in range(len(parts)):
            d = parts_times[_s]
            l.append((parts[_s], parts_cumulative_times[_s] - d, d))

    last_step = max([t[1] for t in l])
    last_step_dur = max([t[2] for t in l if t[1] == last_step])
    end_time = last_step + last_step_dur

    r = make_plot_json(l)
    """
    # write out the json + javascript
    with open("notesOnlyJSON.js", "w") as f:
        f.write(r)
    """

    report_dir = "test_report"
    if not os.path.exists(report_dir):
        os.mkdir(report_dir)

    # write out the index.html with minor modifications for lane names
    w = make_website_string(javascript_note_data_string=r, end_time=end_time)
    with open(report_dir + os.sep + "report1.html", "w") as f:
        f.write(w)

    # test example, every note 
    l = []
    start = 0.
    for i in range(20, 108):
        l.append((i, start, 1.))
        start += 1.

    last_step = max([t[1] for t in l])
    last_step_dur = max([t[2] for t in l if t[1] == last_step])
    end_time = last_step + last_step_dur
    r = make_plot_json(l)

    w = make_website_string(javascript_note_data_string=r, end_time=end_time)
    with open(report_dir + os.sep + "report2.html", "w") as f:
        f.write(w)

    # no report dir because the load is relative to the index file we write out!
    w = make_index_html_string(["report1", "report2"])
    #w = make_index_html_string(["report2"])
    with open(report_dir + os.sep + "index.html", "w") as f:
        f.write(w)


if __name__ == "__main__":
    test_example()
