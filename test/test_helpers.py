import random
import unittest

from extpt.helpers import get_clip_start_frames


class TestClipStartFrames(unittest.TestCase):

    n_iters = 10_000
    short_vid_frames = 20
    long_vid_frames = 200
    clip_len = 15

    def setUp(self) -> None:
        self.clip_list = [random.randint(self.short_vid_frames, self.long_vid_frames) for _ in range(self.n_iters)]
    
    def test_video_bounds(self):
        all_ok = True
        failure = ""
        for t_frames in self.clip_list:
            s1, s2 = get_clip_start_frames(total_frames=t_frames, clip_length=self.clip_len)
            if s1 > s2:
                all_ok = False
                failure = "Second clip starts before first"
                break
            elif s1 < 0 or s2 < 0:
                all_ok = False
                failure = "Clip start is negative"
                break
            elif s1 > t_frames or s2 > t_frames:
                all_ok = False
                failure = "Clip start is > video frames"
                break
            elif (s1 + self.clip_len) > t_frames or (s2 + self.clip_len) > t_frames:
                all_ok = False
                failure = "Clip end is > video frames"
                break

        self.assertTrue(all_ok, msg=failure)
    
    def test_frames_eq_vidlen(self):
        s1, s2 = get_clip_start_frames(total_frames=self.short_vid_frames, clip_length=self.short_vid_frames)
        self.assertEqual(s1, s2)

        

if __name__ == "__main__":
    unittest.main()