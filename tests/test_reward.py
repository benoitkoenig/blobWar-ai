import unittest

from blob_war_ai import reward

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

class TestRewardMethods(unittest.TestCase):
    def test_calc_kills_no_kill(self):
        old_s = generate_state()
        new_s = generate_state()
        allies_killed, enemies_killed = reward.calc_kills(old_s, new_s)
        self.assertEqual(allies_killed, 0)
        self.assertEqual(enemies_killed, 0)

    def test_calc_kills_proper_kill(self):
        old_s = generate_state()
        new_s = generate_state()
        old_s["army"][0] = generate_blob(.2, .1, 1, alive=False)
        new_s["army"][0] = generate_blob(.2, .1, 1, alive=False)
        new_s["enemy"][2] = generate_blob(.2, .9, 1, alive=False)
        allies_killed, enemies_killed = reward.calc_kills(old_s, new_s)
        self.assertEqual(allies_killed, 0)
        self.assertEqual(enemies_killed, 1)

    def test_calc_kills_kamikaze(self):
        old_s = generate_state()
        new_s = generate_state()
        new_s["army"][0] = generate_blob(.2, .1, 1, alive=False)
        new_s["enemy"][2] = generate_blob(.2, .9, 1, alive=False)
        allies_killed, enemies_killed = reward.calc_kills(old_s, new_s)
        self.assertEqual(allies_killed, 1)
        self.assertEqual(enemies_killed, 1)

    def test_calc_kills_multi_kill(self):
        old_s = generate_state()
        new_s = generate_state()
        new_s["army"][0] = generate_blob(.2, .1, 1, alive=False)
        new_s["enemy"][1] = generate_blob(.2, .5, 1, alive=False)
        new_s["enemy"][2] = generate_blob(.2, .9, 1, alive=False)
        allies_killed, enemies_killed = reward.calc_kills(old_s, new_s)
        self.assertEqual(allies_killed, 1)
        self.assertEqual(enemies_killed, 2)

    def test_calc_end_bonus_not_end_of_game(self):
        end_bonus = reward.calc_end_bonus({"type": "update"})
        self.assertEqual(end_bonus, 0)

    def test_calc_end_bonus_victory(self):
        end_bonus = reward.calc_end_bonus({"type": "endOfGame", "value": "Victory !"})
        self.assertEqual(end_bonus, 5)

    def test_calc_end_bonus_defeat(self):
        end_bonus = reward.calc_end_bonus({"type": "endOfGame", "value": "Defeat"})
        self.assertEqual(end_bonus, 2)

    def test_calc_end_bonus_timeout(self):
        end_bonus = reward.calc_end_bonus({"type": "endOfGame", "value": "Timeout"})
        self.assertEqual(end_bonus, -10)

    def test_calc_end_bonus_draw(self):
        end_bonus = reward.calc_end_bonus({"type": "endOfGame", "value": "Draw"})
        self.assertEqual(end_bonus, 4)

    def test_forbidden_move_allowed_move(self):
        s = generate_state()
        result = reward.forbidden_move(s, 0)
        self.assertEqual(result, 0)

    def test_forbidden_move_dead_blob(self):
        s = generate_state()
        s["army"][0] = generate_blob(.2, .1, 1, alive=False)
        result = reward.forbidden_move(s, 0)
        self.assertEqual(result, 1)

    def test_forbidden_move_unavailable_card(self):
        s = generate_state()
        s["cards"]["availability"][0] = False
        result = reward.forbidden_move(s, 1)
        self.assertEqual(result, 1)

    def test_determine_reward_initial_state(self):
        new_s = generate_state()
        result = reward.determine_reward(None, new_s, None)
        self.assertIsNone(result)

    def test_determine_reward_base_value(self):
        old_s = generate_state()
        new_s = generate_state()
        result = reward.determine_reward(old_s, new_s, 0)
        self.assertEqual(result, -1)

if __name__ == "__main__":
    unittest.main()
