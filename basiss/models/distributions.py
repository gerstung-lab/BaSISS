import pymc3 as pm


class BetaSum(pm.Beta):
    def __init__(self, n=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n

    def logp(self, value):
        return super().logp(value) * self.n
