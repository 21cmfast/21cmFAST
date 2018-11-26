from cosmoHammer import util as _util


class Params(_util.Params):
    def items(self):
        for k, v in zip(self.keys, self.values):
            yield k, v
