from kkpthlib.datasets import check_fetch_jsb_chorales
from kkpthlib.datasets import music_json_to_midi
import os


if __name__ == "__main__":
    # example writing out the midi version 
    jsb_dataset_path = check_fetch_jsb_chorales()
    for ex_name in sorted(os.listdir(jsb_dataset_path)):
        print("writing file {}".format(ex_name.replace(".json", "")))
        opath = "jsb_midi"
        if not os.path.exists(opath):
            os.mkdir(opath)
        read_ex = jsb_dataset_path + os.sep + ex_name
        out_ex = opath + os.sep + ex_name.replace(".json", ".midi")
        music_json_to_midi(read_ex, out_ex)
