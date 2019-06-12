from cosmoHammer import util as _util


class Params(_util.Params):
    def items(self):
        for k, v in zip(self.keys, self.values):
            yield k, v

    def __eq__(self, other):
        if self.__class__.__name__ != other.__class__.__name__:
            return  False

        for i, (k,v) in enumerate(self.items()):
            if k not in other.keys:
                return False
            for j, val in enumerate(v):
                if val != other.values[i][j]:
                    return False

        return True
