import math
import unittest

from blob_war_ai import actions

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
        "type": "update",
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

class TestActionsMethods(unittest.TestCase):
    def check_data(self, action_id, expected_id_blob, expected_id_other_blob, expected_id_action, expected_id_card):
        id_blob, id_other_blob, id_action, id_card = actions.get_action_data(action_id)
        self.assertEqual(id_blob, expected_id_blob)
        self.assertEqual(id_other_blob, expected_id_other_blob)
        self.assertEqual(id_action, expected_id_action)
        self.assertEqual(id_card, expected_id_card)        

    def test_first_blob_moves_to_first_enemy_data(self):
        self.check_data(0, 0, 0, 0, None)

    def test_first_blob_moves_to_first_enemy(self):
        s = generate_state()
        action = actions.get_action(s, 0)
        self.assertEqual(len(action), 1)
        self.assertEqual(action[0]["type"], "server/setDestination")
        self.assertEqual(action[0]["idBlob"], 0)
        self.assertEqual(action[0]["destination"], {"x": .2, "y": 2.1})

    def test_second_blob_card_0_to_second_enemy_data(self):
        self.check_data(13, 1, 1, 1, 0)

    def test_second_blob_card_0_to_second_enemy(self):
        s = generate_state()
        action = actions.get_action(s, 13)
        self.assertEqual(len(action), 2)
        self.assertEqual(action[0]["type"], "server/setDestination")
        self.assertEqual(action[0]["idBlob"], 1)
        self.assertEqual(action[0]["destination"], {"x": .5, "y": 2.1})
        self.assertEqual(action[1]["type"], "server/triggerCard")
        self.assertEqual(action[1]["idBlob"], 1)
        self.assertEqual(action[1]["destination"], {"x": .5, "y": 2.1})
        self.assertEqual(action[1]["idCard"], 0)

    def test_third_blob_card_1_to_third_enemy_data(self):
        self.check_data(26, 2, 2, 2, 1)

    def test_third_blob_card_1_to_third_enemy(self):
        s = generate_state()
        action = actions.get_action(s, 26)
        self.assertEqual(len(action), 2)
        self.assertEqual(action[0]["type"], "server/setDestination")
        self.assertEqual(action[0]["idBlob"], 2)
        self.assertEqual(action[0]["destination"], {"x": .8, "y": 2.1})
        self.assertEqual(action[1]["type"], "server/triggerCard")
        self.assertEqual(action[1]["idBlob"], 2)
        self.assertEqual(action[1]["destination"], {"x": .8, "y": 2.1})
        self.assertEqual(action[1]["idCard"], 1)

if __name__ == "__main__":
    unittest.main()
