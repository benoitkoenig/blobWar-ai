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
            generate_blob(.9, .1, 1),
        ],
        "enemy": [
            generate_blob(.2, .9, 1),
            generate_blob(.5, .9, 1),
            generate_blob(.9, .9, 1),
        ]
    }

class TestStateUtilityMethods(unittest.TestCase):
    def test_get_scalar(self):
        v1 = [0, 1]
        v2 = [1, 2]
        scalar = state.get_scalar(v1, v2)
        self.assertEqual(scalar, 2 / math.sqrt(5))

    def test_get_angle(self):
        b1 = {"x": 0, "y": 1}
        b2 = {"x": 1, "y": 2}
        angle = state.get_angle(b1, b2)
        self.assertAlmostEqual(angle, math.pi / 4)

    def test_get_distance(self):
        b1 = {"x": 0, "y": 1}
        b2 = {"x": 1, "y": 2}
        dist = state.get_distance(b1, b2)
        self.assertEqual(dist, 1)

class TestStateGetStateVector(unittest.TestCase):
    def test_1_constant(self):
        s = generate_state()
        state_vector = state.get_state_vector(s, "")
        self.assertEquals(state_vector[0], 1)

    def test_cards_availability(self):
        s = generate_state()
        state_vector = state.get_state_vector(s, "")
        self.assertEqual(state_vector[1], True)
        self.assertEqual(state_vector[2], True)
        s["cards"]["availability"][0] = False
        state_vector = state.get_state_vector(s, "")
        self.assertEqual(state_vector[1], False)
        self.assertEqual(state_vector[2], True)
        s["cards"]["availability"][1] = False
        state_vector = state.get_state_vector(s, "")
        self.assertEqual(state_vector[1], False)
        self.assertEqual(state_vector[2], False)

if __name__ == "__main__":
    unittest.main()
