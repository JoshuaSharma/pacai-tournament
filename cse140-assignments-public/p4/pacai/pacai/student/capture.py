import typing
import pacai.core.action
import pacai.core.agent
import pacai.core.agentinfo
import pacai.capture.gamestate
import pacai.search.distance


def create_team() -> list[pacai.core.agentinfo.AgentInfo]:
    offensive_info = pacai.core.agentinfo.AgentInfo(
        name=f"{__name__}.OffensiveCaptureAgent")
    defensive_info = pacai.core.agentinfo.AgentInfo(
        name=f"{__name__}.DefensiveCaptureAgent")
    return [offensive_info, defensive_info]


class BaseCaptureAgent(pacai.core.agent.Agent):
    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        self._distance_pre = None

    def game_start(self, initial_state) -> None:
        if self._distance_pre is None:
            self._distance_pre = pacai.search.distance.DistancePreComputer()
            self._distance_pre.compute(initial_state.board)
        super().game_start(initial_state)

    def _maze_distance(self, a, b, default: float = 9999.0) -> float:
        if self._distance_pre is not None:
            return self._distance_pre.get_distance_default(a, b, default)
        return pacai.search.distance.manhattan_distance(a, b)

    def get_action(self, state: pacai.capture.gamestate.GameState) -> pacai.core.action.Action:
        legal_actions = state.get_legal_actions()
        if not legal_actions:
            return pacai.core.action.STOP
        values = []
        for action in legal_actions:
            values.append(self.evaluate(state, action))
        best_value = max(values)
        best_actions = []
        for action, value in zip(legal_actions, values):
            if value == best_value:
                best_actions.append(action)
        return self.rng.choice(best_actions)

    def evaluate(self, state, action):
        raise NotImplementedError


class OffensiveCaptureAgent(BaseCaptureAgent):
    _SCORE_WEIGHT = 12.0
    _FOOD_DIST_WEIGHT = 1.0
    _GHOST_DANGER_WEIGHT = 35.0
    _STOP_PENALTY = 10.0
    _PACMAN_BONUS = 5.0
    _DANGER_RADIUS = 3.0

    def evaluate(self, state, action):
        successor = state.generate_successor(action)
        my_pos = successor.get_agent_position(self.agent_index)
        if my_pos is None:
            return -1e6
        score = successor.get_normalized_score(self.agent_index)
        food_positions = successor.get_food(agent_index=self.agent_index)
        if food_positions:
            dists = []
            for fp in food_positions:
                dists.append(self._maze_distance(my_pos, fp, default=9999.0))
            min_food_dist = min(dists)
        else:
            min_food_dist = 0.0
        is_pacman = successor.is_pacman(self.agent_index)
        is_scared = successor.is_scared(self.agent_index)
        nonscared_opponents = successor.get_nonscared_opponent_positions(
            self.agent_index)
        if is_pacman and nonscared_opponents:
            gdists = []
            for pos in nonscared_opponents.values():
                gdists.append(self._maze_distance(my_pos, pos, default=9999.0))
            min_ghost_dist = min(gdists)
        else:
            min_ghost_dist = 9999.0
        d = 0.0
        if is_pacman and not is_scared and min_ghost_dist < self._DANGER_RADIUS:
            d = (self._DANGER_RADIUS - min_ghost_dist) * self._GHOST_DANGER_WEIGHT
        stop_penalty = self._STOP_PENALTY if action == pacai.core.action.STOP else 0.0
        value = self._SCORE_WEIGHT * score - self._FOOD_DIST_WEIGHT * float(min_food_dist) - d - stop_penalty
        if is_pacman:
            value += self._PACMAN_BONUS
        return value


class DefensiveCaptureAgent(BaseCaptureAgent):
    _SCORE_WEIGHT = 5.0
    _INVADER_COUNT_WEIGHT = 100.0
    _INVADER_DIST_WEIGHT = 5.0
    _STOP_PENALTY = 5.0
    _CROSS_PENALTY = 20.0

    def evaluate(self, state, action):
        successor = state.generate_successor(action)
        my_pos = successor.get_agent_position(self.agent_index)
        if my_pos is None:
            return -1e6
        score = successor.get_normalized_score(self.agent_index)
        invaders = successor.get_invader_positions(self.agent_index)
        num_invaders = len(invaders)
        if num_invaders > 0:
            dists = []
            for pos in invaders.values():
                dists.append(self._maze_distance(my_pos, pos, default=9999.0))
            min_invader_dist = min(dists)
        else:
            min_invader_dist = 0.0
        is_pacman = successor.is_pacman(self.agent_index)
        stop_penalty = self._STOP_PENALTY if action == pacai.core.action.STOP else 0.0
        cross_penalty = 0.0
        if is_pacman and num_invaders == 0:
            cross_penalty = self._CROSS_PENALTY
        value = self._SCORE_WEIGHT * score
        if num_invaders > 0:
            value -= self._INVADER_COUNT_WEIGHT * float(num_invaders)
            value -= self._INVADER_DIST_WEIGHT * float(min_invader_dist)
            if not is_pacman:
                value += 20.0
        value -= stop_penalty
        value -= cross_penalty
        return value
