class LazyContextManager:
    class Proxy:
        def __init__(self, lcm):
            self.lcm = lcm

        def __getattr__(self, item):
            return getattr(self.lcm.real_cm, item)

    def __init__(self, class_, *args, **kwargs):
        self.class_ = class_
        self.args = args
        self.kwargs = kwargs

        self.instance = None
        self.cm = None

    def __enter__(self):
        return self.Proxy(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.cm:
            return
        return self.real_cm.__exit__(exc_type, exc_val, exc_tb)

    @property
    def real_cm(self):
        if not self.cm:
            self.instance = self.class_(*self.args, **self.kwargs)
            self.cm = self.instance.__enter__()
        return self.cm
