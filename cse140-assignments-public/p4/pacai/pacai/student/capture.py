import typing

import pacai.core.action
import pacai.core.agent
import pacai.core.agentinfo
import pacai.capture.gamestate
import pacai.search.distance

TEAM_MEMORY: dict[str, int] = {
    "respawn_window_timer": 0,
}

_RESPAWN_WINDOW_LENGTH = 10


def create_team() -> list[pacai.core.agentinfo.AgentInfo]:
    offensive_info = pacai.core.agentinfo.AgentInfo(
        name=f"{__name__}.OffensiveCaptureAgent"
    )
    defensive_info = pacai.core.agentinfo.AgentInfo(
        name=f"{__name__}.DefensiveCaptureAgent"
    )
    return [offensive_info, defensive_info]


class BaseCaptureAgent(pacai.core.agent.Agent):

    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        self._distance_pre: pacai.search.distance.DistancePreComputer | None = None
        self._team_modifier: int = 0

    def game_start(self, initial_state) -> None:
        if self._distance_pre is None:
            self._distance_pre = pacai.search.distance.DistancePreComputer()
            self._distance_pre.compute(initial_state.board)
        self._team_modifier = -1 if (self.agent_index % 2 == 0) else 1
        super().game_start(initial_state)

    def _maze_distance(self, a, b, default: float = 9999.0) -> float:
        if self._distance_pre is not None:
            return self._distance_pre.get_distance_default(a, b, default)
        return pacai.search.distance.manhattan_distance(a, b)

    def get_action(
        self,
        state: pacai.capture.gamestate.GameState
    ) -> pacai.core.action.Action:
        legal_actions = state.get_legal_actions()
        if not legal_actions:
            return pacai.core.action.STOP
        values: list[float] = []
        for action in legal_actions:
            values.append(self.evaluate(state, action))
        best_value = max(values)
        candidate_actions: list[pacai.core.action.Action] = []
        for action, value in zip(legal_actions, values):
            if value == best_value:
                candidate_actions.append(action)
        chosen = self.rng.choice(candidate_actions)
        if TEAM_MEMORY["respawn_window_timer"] > 0:
            TEAM_MEMORY["respawn_window_timer"] -= 1
        succ = state.generate_successor(chosen)
        before_inv = state.get_invader_positions(self.agent_index)
        after_inv = succ.get_invader_positions(self.agent_index)
        if len(after_inv) < len(before_inv):
            TEAM_MEMORY["respawn_window_timer"] = max(
                TEAM_MEMORY["respawn_window_timer"],
                _RESPAWN_WINDOW_LENGTH,
            )
        return chosen

    def evaluate(
        self,
        state: pacai.capture.gamestate.GameState,
        action: pacai.core.action.Action,
    ) -> float:
        raise NotImplementedError


class OffensiveCaptureAgent(BaseCaptureAgent):
    _SCORE_WEIGHT = 15.0
    _FOOD_DIST_WEIGHT = 1.0
    _FOOD_EATEN_WEIGHT = 22.0
    _GHOST_DANGER_WEIGHT = 40.0
    _DANGER_RADIUS = 3.5
    _OFF_INVADER_COUNT_WEIGHT = 80.0
    _OFF_INVADER_DIST_WEIGHT = 4.0
    _STOP_PENALTY = 10.0
    _PACMAN_BONUS = 6.0
    _STEP_PENALTY = 0.1
    _REVERSE_PENALTY = 2.0

    def evaluate(self, state, action):
        successor: pacai.capture.gamestate.GameState = state.generate_successor(action)
        my_pos = successor.get_agent_position(self.agent_index)
        if my_pos is None:
            return -1e6
        score = successor.get_normalized_score(self.agent_index)
        food_positions = successor.get_food(agent_index=self.agent_index)
        if food_positions:
            min_food_dist = min(
                self._maze_distance(my_pos, fp, default=9999.0)
                for fp in food_positions
            )
        else:
            min_food_dist = 0.0
        cur_food_left = state.food_count(agent_index=self.agent_index)
        next_food_left = successor.food_count(agent_index=self.agent_index)
        food_eaten = max(0, cur_food_left - next_food_left)
        is_pacman = successor.is_pacman(self.agent_index)
        is_scared = successor.is_scared(self.agent_index)
        nonscared_opponents = successor.get_nonscared_opponent_positions(
            self.agent_index
        )
        scared_opponents = successor.get_scared_opponent_positions(
            self.agent_index
        )
        invaders = successor.get_invader_positions(self.agent_index)
        respawn_window = TEAM_MEMORY.get("respawn_window_timer", 0) > 0
        min_ghost_dist = 9999.0
        if is_pacman and nonscared_opponents:
            min_ghost_dist = min(
                self._maze_distance(my_pos, pos, default=9999.0)
                for pos in nonscared_opponents.values()
            )
        danger_penalty = 0.0
        if is_pacman and (not is_scared) and (min_ghost_dist < self._DANGER_RADIUS):
            ghost_weight = self._GHOST_DANGER_WEIGHT
            if respawn_window:
                ghost_weight *= 0.4
            danger_penalty = (self._DANGER_RADIUS - min_ghost_dist) * ghost_weight
        scared_bonus = 0.0
        if is_pacman and scared_opponents:
            min_scared = min(
                self._maze_distance(my_pos, pos, default=9999.0)
                for pos in scared_opponents.values()
            )
            scared_bonus = 1.0 / (1.0 + float(min_scared))
        off_defense_penalty = 0.0
        num_invaders = len(invaders)
        if (not is_pacman) and num_invaders > 0:
            invader_min_dist = min(
                self._maze_distance(my_pos, pos, default=9999.0)
                for pos in invaders.values()
            )
            weight_scale = 0.4 if respawn_window else 1.0
            off_defense_penalty += weight_scale * (
                self._OFF_INVADER_COUNT_WEIGHT * float(num_invaders)
                + self._OFF_INVADER_DIST_WEIGHT * float(invader_min_dist)
            )
        stop_penalty = self._STOP_PENALTY if action == pacai.core.action.STOP else 0.0
        last_action = state.get_last_agent_action(self.agent_index)
        reverse_penalty = 0.0
        if last_action is not None:
            rev = pacai.core.action.get_reverse_direction(last_action)
            if rev is not None and action == rev:
                reverse_penalty = self._REVERSE_PENALTY
        value = 0.0
        value += self._SCORE_WEIGHT * score
        value -= self._FOOD_DIST_WEIGHT * float(min_food_dist)
        value -= danger_penalty
        value -= off_defense_penalty
        value -= stop_penalty
        value -= reverse_penalty
        value -= self._STEP_PENALTY
        if is_pacman:
            value += self._PACMAN_BONUS
            if respawn_window:
                value += 3.0
        if food_eaten > 0:
            value += self._FOOD_EATEN_WEIGHT * float(food_eaten)
        value += scared_bonus
        return value


class DefensiveCaptureAgent(BaseCaptureAgent):
    _SCORE_WEIGHT = 6.0
    _INVADER_COUNT_WEIGHT = 120.0
    _INVADER_DIST_WEIGHT = 7.0
    _HOME_FOOD_DIST_WEIGHT = 0.6
    _PATROL_DIST_WEIGHT = 1.0
    _STOP_PENALTY = 6.0
    _CROSS_PENALTY = 22.0
    _STEP_PENALTY = 0.1
    _REVERSE_PENALTY = 2.0

    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        self._patrol_target = None

    def _update_patrol_target(self, state: pacai.capture.gamestate.GameState) -> None:
        home_food = state.get_food(team_modifier=-self._team_modifier)
        if not home_food:
            self._patrol_target = None
            return
        rows = [p.row for p in home_food]
        cols = [p.col for p in home_food]
        avg_r = sum(rows) / float(len(rows))
        avg_c = sum(cols) / float(len(cols))

        def sq_dist(p):
            dr = p.row - avg_r
            dc = p.col - avg_c
            return dr * dr + dc * dc

        self._patrol_target = min(home_food, key=sq_dist)

    def get_action(
        self,
        state: pacai.capture.gamestate.GameState
    ) -> pacai.core.action.Action:
        self._update_patrol_target(state)
        return super().get_action(state)

    def evaluate(self, state, action):
        successor: pacai.capture.gamestate.GameState = state.generate_successor(action)
        my_pos = successor.get_agent_position(self.agent_index)
        if my_pos is None:
            return -1e6
        score = successor.get_normalized_score(self.agent_index)
        invaders = successor.get_invader_positions(self.agent_index)
        num_invaders = len(invaders)
        if num_invaders > 0:
            min_invader_dist = min(
                self._maze_distance(my_pos, pos, default=9999.0)
                for pos in invaders.values()
            )
        else:
            min_invader_dist = 0.0
        is_pacman = successor.is_pacman(self.agent_index)
        is_scared = successor.is_scared(self.agent_index)
        home_food_positions = successor.get_food(
            team_modifier=-self._team_modifier
        )
        if home_food_positions:
            home_food_min_dist = min(
                self._maze_distance(my_pos, fp, default=9999.0)
                for fp in home_food_positions
            )
        else:
            home_food_min_dist = 0.0
        stop_penalty = self._STOP_PENALTY if action == pacai.core.action.STOP else 0.0
        cross_penalty = 0.0
        if is_pacman and num_invaders == 0:
            cross_penalty = self._CROSS_PENALTY
        value = 0.0
        value += self._SCORE_WEIGHT * score
        if num_invaders > 0:
            dist_weight = self._INVADER_DIST_WEIGHT * (0.5 if is_scared else 1.0)
            value -= self._INVADER_COUNT_WEIGHT * float(num_invaders)
            value -= dist_weight * float(min_invader_dist)
            if not is_pacman:
                value += 25.0
        else:
            if not is_pacman:
                value -= self._HOME_FOOD_DIST_WEIGHT * float(home_food_min_dist)
            if (not is_pacman) and (self._patrol_target is not None):
                patrol_dist = self._maze_distance(
                    my_pos,
                    self._patrol_target,
                    default=9999.0,
                )
                value -= self._PATROL_DIST_WEIGHT * float(patrol_dist)
        last_action = state.get_last_agent_action(self.agent_index)
        reverse_penalty = 0.0
        if last_action is not None:
            rev = pacai.core.action.get_reverse_direction(last_action)
            if rev is not None and action == rev:
                reverse_penalty = self._REVERSE_PENALTY
        value -= stop_penalty
        value -= cross_penalty
        value -= reverse_penalty
        value -= self._STEP_PENALTY
        return value
