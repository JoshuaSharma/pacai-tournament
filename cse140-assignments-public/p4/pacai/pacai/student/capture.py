import typing
import pacai.core.action
import pacai.core.agent
import pacai.core.agentinfo
import pacai.capture.gamestate
import pacai.search.distance

def create_team() -> list[pacai.core.agentinfo.AgentInfo]:
    offensive_info = pacai.core.agentinfo.AgentInfo(name=f"{__name__}.OffensiveCaptureAgent")
    defensive_info = pacai.core.agentinfo.AgentInfo(name=f"{__name__}.DefensiveCaptureAgent")
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
    _SCORE_WEIGHT = 13.0
    _FOOD_DIST_WEIGHT = 1.0
    _GHOST_DANGER_WEIGHT = 35.0
    _STOP_PENALTY = 10.0
    _PACMAN_BONUS = 6.0
    _DANGER_RADIUS = 3.0
    _STEP_PENALTY = 0.1
    _REVERSE_PENALTY = 2.0

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
        nonscared_opponents = successor.get_nonscared_opponent_positions(self.agent_index)
        scared_opponents = successor.get_scared_opponent_positions(self.agent_index)
        if is_pacman and nonscared_opponents:
            gdists = []
            for pos in nonscared_opponents.values():
                gdists.append(self._maze_distance(my_pos, pos, default=9999.0))
            min_ghost_dist = min(gdists)
        else:
            min_ghost_dist = 9999.0
        danger_penalty = 0.0
        if is_pacman and not is_scared and min_ghost_dist < self._DANGER_RADIUS:
            danger_penalty = (self._DANGER_RADIUS - min_ghost_dist) * self._GHOST_DANGER_WEIGHT
        scared_bonus = 0.0
        if is_pacman and scared_opponents:
            sdists = []
            for pos in scared_opponents.values():
                sdists.append(self._maze_distance(my_pos, pos, default=9999.0))
            min_scared = min(sdists)
            scared_bonus = 2.0 / (1.0 + float(min_scared))
        stop_penalty = self._STOP_PENALTY if action == pacai.core.action.STOP else 0.0
        last_action = state.get_last_agent_action(self.agent_index)
        reverse_penalty = 0.0
        if last_action is not None:
            rev = pacai.core.action.get_reverse_direction(last_action)
            if rev is not None and action == rev:
                reverse_penalty = self._REVERSE_PENALTY
        value = self._SCORE_WEIGHT * score - self._FOOD_DIST_WEIGHT * float(min_food_dist) - danger_penalty - \
            stop_penalty - reverse_penalty - self._STEP_PENALTY
        if is_pacman:
            value += self._PACMAN_BONUS
        value += scared_bonus
        return value

class DefensiveCaptureAgent(BaseCaptureAgent):
    _SCORE_WEIGHT = 5.0
    _INVADER_COUNT_WEIGHT = 110.0
    _INVADER_DIST_WEIGHT = 6.0
    _STOP_PENALTY = 6.0
    _CROSS_PENALTY = 22.0
    _STEP_PENALTY = 0.1
    _REVERSE_PENALTY = 2.0

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
        is_scared = successor.is_scared(self.agent_index)
        stop_penalty = self._STOP_PENALTY if action == pacai.core.action.STOP else 0.0
        cross_penalty = 0.0
        if is_pacman and num_invaders == 0:
            cross_penalty = self._CROSS_PENALTY
        value = self._SCORE_WEIGHT * score
        if num_invaders > 0:
            dist_weight = self._INVADER_DIST_WEIGHT * (0.5 if is_scared else 1.0)
            value -= self._INVADER_COUNT_WEIGHT * float(num_invaders)
            value -= dist_weight * float(min_invader_dist)
            if not is_pacman:
                value += 25.0
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
