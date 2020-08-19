import numpy as np

class MuEncoder(object):
    def __init__(self, datarange, mu=255):
        self.mu = mu
        self.datarange = datarange

    def normalize(self, x, span=None):
        """
        Scale x from range of span to range [-1, 1]

        The span can be specified using a string argument or using a length-2 
        list-like. If not specified, normalize attempts to use self.datarange as 
        span. If self.datarange is also None, then the min and max of x are used.
        """

        def _span_datarange(datarange):
            lower = datarange[0]
            span = datarange[1] - lower
            return (span, lower)

        def _span_minmax(x):
            lower = np.min(x)
            span = np.max(x) - lower
            return (span, lower)

        if span is None:
            if self.datarange is not None:
                span, lower = _span_datarange(self.datarange)
            else:
                span, lower = _span_minmax(x)
        elif isinstance(span, str):
            if span == 'datarange':
                span, lower = _span_datarange(self.datarange)
            elif span == 'minmax':
                span, lower = _span_minmax(x)
            else:
                span, lower = _span_minmax(x)
        else:
            lower = span[0]
            span = span[1] - lower

        return ((np.float32(x) - lower) / span - 0.5) * 2

    def expand(self, x):
        """
        Scale x from range of [-1, 1] to self.datarange
        """

        span = self.datarange[1] - self.datarange[0]
        return (x / 2 + 0.5) * span + self.datarange[0]

    def encode(self, x):
        x = self.normalize(x)
        return np.sign(x) * np.log(1 + self.mu * np.abs(x)) / np.log(1 + self.mu)

    def decode(self, x):
        x = np.sign(x) * self.mu**-1 * ((1 + self.mu)**np.abs(x) - 1)
        return self.expand(x)
