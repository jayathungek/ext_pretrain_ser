import unittest


class Counter:
    def __init__(self, start_val=0) -> None:
        self.count = start_val

    def inc(self, inc_val=1):
        self.count += inc_val


class TestModuleImports(unittest.TestCase):
    """
    This tests that all the non-builtin modules used in the project
    are able to load correctly. Useful when running for the first
    time on a new podman container to see whether the environment
    was set up correctly.
    """
    def setUp(self) -> None:
        self.raised = False
        self.msg = ""
        self.failed_line = None
        self.line_counter = Counter(start_val=1)
    
    def tearDown(self) -> None:
        if self.raised:
            self.fail(f"Import error on line {self.failed_line}: {self.msg}")

    def test_torch_cuda(self):
        try:
            import torch; self.line_counter.inc()
            import torchvision; self.line_counter.inc()
            import torchaudio; self.line_counter.inc()
            import torchyin; self.line_counter.inc()

        except ImportError as e:
            self.raised = True
            self.msg = e
            self.failed_line = self.line_counter.count

        self.assertTrue(torch.cuda.is_available())

    def test_clipmbt(self):
        try:
            import extpt; self.line_counter.inc()
            import extpt.augment; self.line_counter.inc()
            import extpt.datasets; self.line_counter.inc()
            import extpt.open_clip; self.line_counter.inc()
            import extpt.train; self.line_counter.inc()
            import extpt.helpers; self.line_counter.inc()
            import extpt.tokenizer; self.line_counter.inc()
            import extpt.data; self.line_counter.inc()
            import extpt.clip; self.line_counter.inc()
            import extpt.loss; self.line_counter.inc()
            import extpt.metrics; self.line_counter.inc()
            import extpt.transformer; self.line_counter.inc()
            import extpt.visualise; self.line_counter.inc()
            import extpt.vit; self.line_counter.inc()
            
        except ImportError as e:
            self.raised = True
            self.msg = e
            self.failed_line = self.line_counter.count
    
    def test_notebook_imports(self):
        try:
            import matplotlib.pyplot as plt; self.line_counter.inc()
            import numpy as np; self.line_counter.inc()
            import pandas as pd; self.line_counter.inc()
            from ipywidgets import HBox; self.line_counter.inc()
            from IPython.display import display; self.line_counter.inc()
            import plotly.io as pio; self.line_counter.inc()
            import plotly.graph_objects as go; self.line_counter.inc()
            from plotly.offline import init_notebook_mode; self.line_counter.inc()

        except ImportError as e:
            self.raised = True
            self.msg = e
            self.failed_line = self.line_counter.count
        

if __name__ == "__main__":
    unittest.main()