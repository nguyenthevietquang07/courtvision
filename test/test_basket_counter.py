import unittest

from scripts.detect_crossings import BasketCounter


NET_BOX = (100.0, 100.0, 140.0, 150.0)


class BasketCounterTests(unittest.TestCase):
    def test_counts_downward_rim_crossing(self) -> None:
        counter = BasketCounter()

        self.assertEqual(counter.update(10, (120.0, 90.0), NET_BOX), (False, "armed_above_rim"))
        self.assertEqual(counter.update(14, (120.0, 110.0), NET_BOX), (True, "above_to_below_rim"))

    def test_does_not_count_ball_that_starts_below_rim(self) -> None:
        counter = BasketCounter()

        self.assertEqual(counter.update(10, (120.0, 110.0), NET_BOX), (False, ""))

    def test_does_not_count_expired_transition(self) -> None:
        counter = BasketCounter(max_transition_frames=5)

        counter.update(10, (120.0, 90.0), NET_BOX)
        self.assertEqual(counter.update(16, (120.0, 110.0), NET_BOX), (False, ""))

    def test_does_not_count_ball_outside_rim_width(self) -> None:
        counter = BasketCounter(horizontal_padding=10.0)

        counter.update(10, (160.0, 90.0), NET_BOX)
        self.assertEqual(counter.update(12, (160.0, 110.0), NET_BOX), (False, ""))

    def test_cooldown_prevents_duplicate_score(self) -> None:
        counter = BasketCounter(cooldown_frames=30)

        counter.update(10, (120.0, 90.0), NET_BOX)
        self.assertTrue(counter.update(12, (120.0, 110.0), NET_BOX)[0])
        counter.update(20, (120.0, 90.0), NET_BOX)
        self.assertEqual(counter.update(22, (120.0, 110.0), NET_BOX), (False, ""))


if __name__ == "__main__":
    unittest.main()
