import math
import unittest

from blob_war_ai import state

def generate_blob(x, y, orientation, alive=True, status="normal", destination=None):
    return {
        "x": x,
        "y": y,
        "orientation": orientation,
        "alive": alive,
        "status": status,
        "destination": destination,
    }

def generate_state():
    return {
        "cards": {
            "availability": [True, True],
            "currentBlob": [None, None],
        },
        "army": [
            generate_blob(.2, .1, 1),
            generate_blob(.5, .1, 1),
            generate_blob(.8, .1, 1),
        ],
        "enemy": [
            generate_blob(.2, .9, 1),
            generate_blob(.5, .9, 1),
            generate_blob(.8, .9, 1),
        ]
    }

class TestStateUtilityMethods(unittest.TestCase):
    def test_get_scalar(self):
        v0 = [0, 0]
        v1 = [0, 1]
        v2 = [1, 2]
        self.assertEqual(state.get_scalar(v0, v2), 0)
        self.assertEqual(state.get_scalar(v1, v2), 2 / math.sqrt(5))

    def test_get_angle(self):
        b1 = {"x": 0, "y": 1}
        b2 = {"x": 1, "y": 2}
        b3 = {"x": 1, "y": 0}
        self.assertAlmostEqual(state.get_angle(b1, b2), math.pi / 4)
        self.assertAlmostEqual(state.get_angle(b1, b3), -math.pi / 4)

    def test_get_distance(self):
        b1 = {"x": 0, "y": 1}
        b2 = {"x": 1, "y": 2}
        dist = state.get_distance(b1, b2)
        self.assertEqual(dist, 1)

class TestStateGetStateMethods(unittest.TestCase):
    def test_get_cards_data(self):
        s = generate_state()
        s["cards"]["availability"][0] = False
        s["cards"]["currentBlob"][0] = 1
        state_vector = state.get_cards_data(s)
        self.assertEqual(state_vector, [0, 1, 0, 0, 1, 0, 0, 0])

    def test_get_all_blobs_features(self):
        s = generate_state()
        s["army"][0] = generate_blob(.2, .1, 1, alive=False)
        s["army"][1] = generate_blob(.5, .1, 1, status="hat")
        s["army"][2] = generate_blob(.8, .1, 1, status="ghost")
        s["enemy"][0] = generate_blob(.2, .9, 3, status="hat")
        state_vector = state.get_all_blobs_features(s)
        self.assertEqual(state_vector[0:6], [0, 1, 0, 0, .2, .1])
        self.assertEqual(state_vector[6:12], [1, 0, 0, 1, .5, .1])
        self.assertEqual(state_vector[12:18], [1, 0, 1, 0, .8, .1])
        self.assertEqual(state_vector[18:24], [1, 0, 0, 1, .2, .9])
        self.assertEqual(state_vector[24:30], [1, 1, 0, 0, .5, .9])
        self.assertEqual(state_vector[30:36], [1, 1, 0, 0, .8, .9])

    def test_get_pair_blob_features(self):
        s = generate_state()
        s["enemy"][0] = generate_blob(.2, .4, 3)
        s["enemy"][1] = generate_blob(.5, .4, 3)
        s["enemy"][2] = generate_blob(.8, .4, 3)
        state_vector = state.get_pair_blob_features(s)
        # 1, 0
        self.assertAlmostEqual(state_vector[0], .3 / math.sqrt(2))
        self.assertAlmostEqual(state_vector[1], math.pi)
        self.assertAlmostEqual(state_vector[2], 0)
        # 2, 1
        self.assertAlmostEqual(state_vector[6], .3 / math.sqrt(2))
        self.assertAlmostEqual(state_vector[7], math.pi)
        self.assertAlmostEqual(state_vector[8], 0)
        # 3, 0
        self.assertAlmostEqual(state_vector[9], .3 / math.sqrt(2))
        self.assertAlmostEqual(state_vector[10], -math.pi / 2)
        self.assertAlmostEqual(state_vector[11], math.pi / 2)
        # 4, 2
        self.assertAlmostEqual(state_vector[24], .3)
        self.assertAlmostEqual(state_vector[25], - math.pi / 4)
        self.assertAlmostEqual(state_vector[26], 3 * math.pi / 4)
        # 5, 0
        self.assertAlmostEqual(state_vector[30], math.sqrt(.45 / 2))
        self.assertAlmostEqual(state_vector[31], -2.677945044588987)
        self.assertAlmostEqual(state_vector[32], math.pi - 2.677945044588987)
        # 5, 4
        self.assertAlmostEqual(state_vector[42], .3 / math.sqrt(2))
        self.assertAlmostEqual(state_vector[43], math.pi)
        self.assertAlmostEqual(state_vector[44], 0)

    def test_get_angle_between_three_blobs(self):
        s = generate_state()
        s["enemy"][0] = generate_blob(.2, .4, 3)
        s["enemy"][1] = generate_blob(.5, .4, 3)
        s["enemy"][2] = generate_blob(.8, .4, 3)
        state_vector = state.get_angle_between_three_blobs(s)
        # 0, 2, 1
        self.assertAlmostEqual(state_vector[0], 0)
        # 1, 2, 0
        self.assertAlmostEqual(state_vector[10], -math.pi)
        # 2, 1, 0
        self.assertAlmostEqual(state_vector[20], 0)
        # 3, 1, 0
        self.assertAlmostEqual(state_vector[30], math.pi / 4)
        # 4, 1, 0
        self.assertAlmostEqual(state_vector[40], math.pi / 4)
        # 5, 1, 0
        self.assertAlmostEqual(state_vector[50], 0.32175055439664213)

    def test_in_range_data(self):
        s = generate_state()
        s["enemy"][0] = generate_blob(.1, .1, 3)
        s["enemy"][1] = generate_blob(.5, .1, 3, status="ghost")
        state_vector = state.get_in_range_data(s, "RBotGhostBloc")
        self.assertEqual(state_vector, [0, 0, 0, 0, 1, 0, 0, 0, 0])
        state_vector = state.get_in_range_data(s, "RBotDashDash")
        self.assertEqual(state_vector, [1, 0, 0, 0, 1, 0, 0, 0, 0])
        state_vector = state.get_in_range_data(s, "ThisOneDoesNotExist")
        self.assertEqual(state_vector, [0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_get_scalar_data(self):
        s = generate_state()
        s["army"][0] = generate_blob(.2, .1, 1, destination={"x": .2, "y": .9})
        s["enemy"][0] = generate_blob(.2, .4, 3)
        s["enemy"][1] = generate_blob(.5, .4, 3)
        s["enemy"][2] = generate_blob(.8, .4, 3)
        state_vector = state.get_scalar_data(s)
        self.assertEqual(state_vector, [1, 1 / math.sqrt(2), 1 / math.sqrt(5), 0, 0, 0, 0, 0, 0])

    def test_state_get_state_vector(self):
        s = generate_state()
        state_vector = state.get_state_vector(s, "")
        self.assertEqual(len(state_vector), 168)
        self.assertEqual(state_vector[0], 1)

if __name__ == "__main__":
    unittest.main()
