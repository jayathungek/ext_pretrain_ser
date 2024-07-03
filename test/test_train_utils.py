import os
import unittest
import shutil
import pathlib

import numpy as np
from extpt.train.train_utils import clean_model_dir


class TestCleanDir(unittest.TestCase):

    def setUp(self) -> None:
        curr_dir = pathlib.Path(__file__).parent.resolve()
        self.test_dir = curr_dir / "example_model_run"
        os.mkdir(self.test_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def _create_files(self, num_files: int):
        lowest = 2
        for _ in range(num_files):
            f = np.random.rand()
            open(self.test_dir / f"example_n_val_loss_{f:.5}.pth", 'a').close() 
            if f < lowest:
                lowest = f
        return lowest
    
    def test_many(self):
        best_loss = self._create_files(100)
        clean_model_dir(self.test_dir, confirmed=True)
        checkpoints = list(self.test_dir.glob("*.pth"))
        self.assertTrue(len(checkpoints) == 1)
        self.assertEqual(checkpoints[0].stem, f"example_n_val_loss_{best_loss:.5}")
        
    def test_one(self):
        best_loss = self._create_files(1)
        clean_model_dir(self.test_dir, confirmed=True)
        checkpoints = list(self.test_dir.glob("*.pth"))
        self.assertTrue(len(checkpoints) == 1)
        self.assertEqual(checkpoints[0].stem, f"example_n_val_loss_{best_loss:.5}")

    def test_empty(self):
        _ = self._create_files(0)
        clean_model_dir(self.test_dir, confirmed=True)
        checkpoints = list(self.test_dir.glob("*.pth"))
        self.assertTrue(len(checkpoints) == 0)


if __name__ == "__main__":
    unittest.main()