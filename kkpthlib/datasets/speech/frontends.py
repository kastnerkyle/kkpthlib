
class EnglishPhonemeLookup(dict):
    def __init__(self):
        super().__init__()
        basedir = "/".join(__file__.split("/")[:-1])
        with open(basedir + "/en_lang/phones.txt", "r") as f:
            lines = f.readlines()
        phone_syms = [l.split(" ")[0] for l in lines]
        for _n, ps in enumerate(phone_syms):
            super().__setitem__(ps, _n)

    def __getattr__(self, item):
        return super().__getitem__(item)

    def __setattr__(self, item, value):
        return super().__setitem__(item, value)
