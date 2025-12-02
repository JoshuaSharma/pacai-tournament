import typing
from collections import deque

import pacai.core.action
import pacai.core.agent
import pacai.core.agentinfo
import pacai.capture.gamestate
import pacai.search.distance
import pacai.core.board

TEAM_MEMORY: dict[str, int] = {
    "respawn_window_timer": 0,
}
_RESPAWN_WINDOW_LENGTH = 10
_DISTANCE_CACHE: dict[str, pacai.search.distance.DistancePreComputer] = {}


def create_team() -> list[pacai.core.agentinfo.AgentInfo]:
    offensive_info = pacai.core.agentinfo.AgentInfo(
        name=f"{__name__}.OffensiveCaptureAgent"
    )
    defensive_info = pacai.core.agentinfo.AgentInfo(
        name=f"{__name__}.DefensiveCaptureAgent"
    )
    return [offensive_info, defensive_info]


class BaseCaptureAgent(pacai.core.agent.Agent):
    _LATE_GAME_TURN = 800

    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        self._distance_pre: pacai.search.distance.DistancePreComputer | None = None
        self._team_modifier: int = 0
        self._board: pacai.core.board.Board | None = None

    @staticmethod
    def _get_turn_count(state: pacai.core.gamestate.GameState) -> int:
        return getattr(state, "turn_count",
                       getattr(state, "turncount", 0))

    def _init_home_border_positions(self, board: pacai.core.board.Board) -> None:
        pass

    def game_start(self, initial_state) -> None:
        TEAM_MEMORY["respawn_window_timer"] = 0
        board = initial_state.board
        self._board = board
        self._team_modifier = -1 if (self.agent_index % 2 == 0) else 1
        board_key = board.get_nonwall_string()
        if board_key in _DISTANCE_CACHE:
            self._distance_pre = _DISTANCE_CACHE[board_key]
        else:
            pre = pacai.search.distance.DistancePreComputer()
            pre.compute(board)
            _DISTANCE_CACHE[board_key] = pre
            self._distance_pre = pre
        self._init_home_border_positions(board)
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
        before_inv = state.get_invader_positions(self.agent_index)
        successors: list[pacai.capture.gamestate.GameState] = []
        values: list[float] = []
        for action in legal_actions:
            succ = state.generate_successor(action)
            successors.append(succ)
            values.append(self._evaluate_successor(state, succ, action))
        best_value = max(values)
        best_indexes = [i for i, v in enumerate(values) if v == best_value]
        chosen_index = self.rng.choice(best_indexes)
        chosen_action = legal_actions[chosen_index]
        chosen_succ = successors[chosen_index]
        if TEAM_MEMORY["respawn_window_timer"] > 0:
            TEAM_MEMORY["respawn_window_timer"] -= 1
        after_inv = chosen_succ.get_invader_positions(self.agent_index)
        if len(after_inv) < len(before_inv):
            TEAM_MEMORY["respawn_window_timer"] = max(
                TEAM_MEMORY["respawn_window_timer"],
                _RESPAWN_WINDOW_LENGTH,
            )
        self._post_action_update(state, chosen_succ)
        return chosen_action

    def evaluate(
        self,
        state: pacai.capture.gamestate.GameState,
        action: pacai.core.action.Action,
    ) -> float:
        successor: pacai.capture.gamestate.GameState = state.generate_successor(action)
        return self._evaluate_successor(state, successor, action)

    def _evaluate_successor(
        self,
        state: pacai.capture.gamestate.GameState,
        successor: pacai.capture.gamestate.GameState,
        action: pacai.core.action.Action,
    ) -> float:
        raise NotImplementedError

    def _post_action_update(
        self,
        state: pacai.capture.gamestate.GameState,
        successor: pacai.capture.gamestate.GameState,
    ) -> None:
        pass


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
    _RETREAT_BORDER_THRESHOLD = 7.0
    _RETREAT_WEIGHT = 3.0

    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        self._home_border_positions: list[pacai.core.board.Position] = []
        self._lane_rows: list[int] = []
        self._lane_row_to_targets: dict[int, list[pacai.core.board.Position]] = {}
        self._lane_index: int = 0
        self._blocked_lanes: set[int] = set()

    def _init_home_border_positions(self, board: pacai.core.board.Board) -> None:
        self._home_border_positions = []
        self._lane_rows = []
        self._lane_row_to_targets = {}
        self._lane_index = 0
        self._blocked_lanes = set()

        height = board.height
        width = board.width
        mid_col = width // 2

        if self._team_modifier == -1:
            home_border_col = mid_col - 1
            enemy_border_col = mid_col
        else:
            home_border_col = mid_col
            enemy_border_col = mid_col - 1

        for row in range(height):
            home_pos = pacai.core.board.Position(row, home_border_col)
            if not board.is_wall(home_pos):
                self._home_border_positions.append(home_pos)

            enemy_pos = pacai.core.board.Position(row, enemy_border_col)
            if not board.is_wall(enemy_pos):
                self._lane_row_to_targets.setdefault(row, []).append(enemy_pos)

        rows = sorted(self._lane_row_to_targets.keys())
        if rows:
            mid_row = rows[len(rows) // 2]
            upper = [r for r in rows if r < mid_row]
            lower = [r for r in rows if r > mid_row]
            self._lane_rows = [mid_row] + upper + lower
        else:
            self._lane_rows = []

    def _bfs_to_target(
        self,
        start: pacai.core.board.Position,
        goal: pacai.core.board.Position,
    ) -> tuple[list[pacai.core.action.Action], list[pacai.core.board.Position]]:
        if self._board is None:
            return [], []
        queue: deque[pacai.core.board.Position] = deque()
        queue.append(start)
        parents: dict[pacai.core.board.Position,
                     tuple[pacai.core.board.Position, pacai.core.action.Action] | None] = {}
        parents[start] = None
        while queue:
            pos = queue.popleft()
            if pos == goal:
                actions: list[pacai.core.action.Action] = []
                positions: list[pacai.core.board.Position] = [pos]
                cur = pos
                while parents[cur] is not None:
                    prev, act = parents[cur]
                    actions.append(act)
                    positions.append(prev)
                    cur = prev
                actions.reverse()
                positions.reverse()
                return actions, positions
            for act, nxt in self._board.get_neighbors(pos):
                if nxt not in parents:
                    parents[nxt] = (pos, act)
                    queue.append(nxt)
        return [], []

    def _forced_cross_action(
        self,
        state: pacai.capture.gamestate.GameState,
        legal_actions: list[pacai.core.action.Action],
    ) -> pacai.core.action.Action | None:
        if self._board is None or not self._lane_rows:
            return None

        my_pos = state.get_agent_position(self.agent_index)
        if my_pos is None:
            return None

        ghosts = state.get_nonscared_opponent_positions(self.agent_index)
        ghost_positions = list(ghosts.values())

        max_tries = len(self._lane_rows)
        for offset in range(max_tries):
            lane_idx = (self._lane_index + offset) % len(self._lane_rows)
            if lane_idx in self._blocked_lanes:
                continue
            row = self._lane_rows[lane_idx]
            targets = self._lane_row_to_targets.get(row, [])
            if not targets:
                self._blocked_lanes.add(lane_idx)
                continue

            target = min(
                targets,
                key=lambda p: self._maze_distance(my_pos, p, default=9999.0)
            )
            path_actions, path_positions = self._bfs_to_target(my_pos, target)
            if not path_actions:
                self._blocked_lanes.add(lane_idx)
                continue

            safe = True
            if ghost_positions:
                steps_to_check = min(6, len(path_positions))
                for i in range(steps_to_check):
                    pos_i = path_positions[i]
                    for gpos in ghost_positions:
                        d = self._maze_distance(pos_i, gpos, default=9999.0)
                        if d <= self._DANGER_RADIUS:
                            safe = False
                            break
                    if not safe:
                        break
            if not safe:
                self._blocked_lanes.add(lane_idx)
                continue

            first_action = path_actions[0]
            if first_action not in legal_actions:
                self._blocked_lanes.add(lane_idx)
                continue

            self._lane_index = lane_idx
            return first_action

        return None

    def get_action(
        self,
        state: pacai.capture.gamestate.GameState
    ) -> pacai.core.action.Action:
        legal_actions = state.get_legal_actions()
        if not legal_actions:
            return pacai.core.action.STOP

        if not state.is_pacman(self.agent_index):
            forced = self._forced_cross_action(state, legal_actions)
            if forced is not None:
                before_inv = state.get_invader_positions(self.agent_index)
                succ = state.generate_successor(forced)
                if TEAM_MEMORY["respawn_window_timer"] > 0:
                    TEAM_MEMORY["respawn_window_timer"] -= 1
                after_inv = succ.get_invader_positions(self.agent_index)
                if len(after_inv) < len(before_inv):
                    TEAM_MEMORY["respawn_window_timer"] = max(
                        TEAM_MEMORY["respawn_window_timer"],
                        _RESPAWN_WINDOW_LENGTH,
                    )
                self._post_action_update(state, succ)
                return forced

        return super().get_action(state)

    def _evaluate_successor(self, state, successor, action):
        my_pos = successor.get_agent_position(self.agent_index)
        if my_pos is None:
            return -1e6

        score = successor.get_normalized_score(self.agent_index)
        turn_count = self._get_turn_count(state)
        late_game = (turn_count >= self._LATE_GAME_TURN)
        behind_or_tied = (score <= 0.0)
        late_aggressive = late_game and behind_or_tied
        food_positions = successor.get_food(agent_index=self.agent_index)
        if food_positions:
            min_food_dist = min(
                self._maze_distance(my_pos, fp, default=9999.0)
                for fp in food_positions
            )
            K = 10
            manhattan_food = sorted(
                food_positions,
                key=lambda fp: pacai.search.distance.manhattan_distance(my_pos, fp)
            )
            candidates = manhattan_food[:K]
            min_food_dist = min(
                self._maze_distance(my_pos, fp, default=9999.0)
                for fp in candidates
            )
            next_food_left = len(food_positions)
        else:
            min_food_dist = 0.0
            next_food_left = 0

        cur_food_left = state.food_count(agent_index=self.agent_index)
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
            elif late_aggressive:
                ghost_weight *= 0.7
            danger_penalty = (self._DANGER_RADIUS - min_ghost_dist) * ghost_weight

        dead_end_penalty = 0.0
        if (
            is_pacman
            and not is_scared
            and nonscared_opponents
            and self._board is not None
        ):
            neighbors = self._board.get_neighbors(my_pos)
            if len(neighbors) <= 1 and min_ghost_dist <= (self._DANGER_RADIUS + 1.0):
                dead_end_penalty = 40.0

        retreat_penalty = 0.0
        if (
            is_pacman
            and score > 0.0
            and min_ghost_dist < (self._DANGER_RADIUS + 1.0)
            and self._home_border_positions
        ):
            border_min_dist = min(
                self._maze_distance(my_pos, bp, default=9999.0)
                for bp in self._home_border_positions
            )
            if border_min_dist > self._RETREAT_BORDER_THRESHOLD:
                retreat_penalty = (
                    border_min_dist - self._RETREAT_BORDER_THRESHOLD
                ) * self._RETREAT_WEIGHT

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
            if late_aggressive:
                off_defense_penalty *= 0.3

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
        value -= dead_end_penalty
        value -= retreat_penalty
        value -= off_defense_penalty
        value -= stop_penalty
        value -= reverse_penalty
        value -= self._STEP_PENALTY

        if is_pacman:
            value += self._PACMAN_BONUS
            if respawn_window:
                value += 3.0
            elif late_aggressive:
                value += 2.0

        if food_eaten > 0:
            value += self._FOOD_EATEN_WEIGHT * float(food_eaten)
        value += scared_bonus
        return value


class DefensiveCaptureAgent(BaseCaptureAgent):
    _SCORE_WEIGHT = 6.0
    _INVADER_COUNT_WEIGHT = 120.0
    _INVADER_DIST_WEIGHT = 7.0
    _HOME_FOOD_DIST_WEIGHT = 0.3
    _PATROL_DIST_WEIGHT = 1.2
    _STOP_PENALTY = 6.0
    _CROSS_PENALTY = 12.0
    _STEP_PENALTY = 0.1
    _REVERSE_PENALTY = 2.0
    _DOUBLE_PACMAN_HOME_FOOD_LIMIT = 3
    _INTRUSION_WEIGHT = 4.0
    _INTRUSION_TTL = 15

    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        self._patrol_target: pacai.core.board.Position | None = None
        self._patrol_targets: list[pacai.core.board.Position] = []
        self._intrusion_targets: list[tuple[pacai.core.board.Position, int]] = []

    def _update_patrol_target(self, state: pacai.capture.gamestate.GameState) -> None:
        home_food = state.get_food(team_modifier=-self._team_modifier)
        self._patrol_targets = []
        if not home_food:
            self._patrol_target = None
            return

        board = state.board
        height = board.height
        center_row = height // 2
        if self._team_modifier == -1:
            extreme_col = max(p.col for p in home_food)
            candidates = sorted(
                [p for p in home_food if p.col == extreme_col],
                key=lambda p: p.row
            )
        else:
            extreme_col = min(p.col for p in home_food)
            candidates = sorted(
                [p for p in home_food if p.col == extreme_col],
                key=lambda p: p.row
            )

        if not candidates:
            self._patrol_target = None
            return

        n = len(candidates)
        if n <= 3:
            self._patrol_targets = candidates
        else:
            top = candidates[0]
            mid = candidates[n // 2]
            bot = candidates[-1]
            self._patrol_targets = [top, mid, bot]

        self._patrol_target = min(
            self._patrol_targets,
            key=lambda p: abs(p.row - center_row),
        )

    def get_action(
        self,
        state: pacai.capture.gamestate.GameState
    ) -> pacai.core.action.Action:
        self._update_patrol_target(state)
        return super().get_action(state)

    def _post_action_update(
        self,
        state: pacai.capture.gamestate.GameState,
        successor: pacai.capture.gamestate.GameState,
    ) -> None:
        new_targets: list[tuple[pacai.core.board.Position, int]] = []
        for pos, ttl in self._intrusion_targets:
            if ttl > 1:
                new_targets.append((pos, ttl - 1))
        self._intrusion_targets = new_targets
        prev_food = state.get_food(team_modifier=-self._team_modifier)
        next_food = successor.get_food(team_modifier=-self._team_modifier)
        eaten = prev_food - next_food
        for pos in eaten:
            self._intrusion_targets.append((pos, self._INTRUSION_TTL))

    def _evaluate_successor(self, state, successor, action):
        my_pos = successor.get_agent_position(self.agent_index)
        if my_pos is None:
            return -1e6

        score = successor.get_normalized_score(self.agent_index)
        turn_count = self._get_turn_count(state)
        late_game = (turn_count >= self._LATE_GAME_TURN)
        behind_or_tied = (score <= 0.0)
        endgame_attack = late_game and behind_or_tied

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
            home_food_left = len(home_food_positions)
        else:
            home_food_min_dist = 0.0
            home_food_left = 0

        ally_positions = successor.get_ally_positions(self.agent_index)
        ally_pacman = any(
            successor.is_pacman(ally_index)
            for ally_index in ally_positions.keys()
        )

        stop_penalty = self._STOP_PENALTY if action == pacai.core.action.STOP else 0.0
        if (
            endgame_attack
            and ally_pacman
            and home_food_left > self._DOUBLE_PACMAN_HOME_FOOD_LIMIT
        ):
            endgame_attack = False

        cross_penalty = 0.0
        if is_pacman and num_invaders == 0:
            if endgame_attack:
                if ally_pacman and home_food_left > self._DOUBLE_PACMAN_HOME_FOOD_LIMIT:
                    cross_penalty = self._CROSS_PENALTY * 1.5
                else:
                    cross_penalty = 0.0
            else:
                if home_food_left > 5:
                    cross_penalty = self._CROSS_PENALTY
                elif home_food_left > 0:
                    cross_penalty = 0.5 * self._CROSS_PENALTY
                else:
                    cross_penalty = 0.0

        value = 0.0
        value += self._SCORE_WEIGHT * score

        if num_invaders > 0:
            dist_weight = self._INVADER_DIST_WEIGHT * (0.5 if is_scared else 1.0)
            value -= self._INVADER_COUNT_WEIGHT * float(num_invaders)
            value -= dist_weight * float(min_invader_dist)
            if not is_pacman:
                value += 25.0
        else:
            home_weight = self._HOME_FOOD_DIST_WEIGHT
            patrol_weight = self._PATROL_DIST_WEIGHT
            if endgame_attack:
                home_weight *= 0.3
                patrol_weight *= 0.3

            if not is_pacman:
                value -= home_weight * float(home_food_min_dist)
            if (not is_pacman) and self._patrol_targets:
                best_patrol = min(
                    self._patrol_targets,
                    key=lambda p: self._maze_distance(my_pos, p, default=9999.0),
                )
                patrol_dist = self._maze_distance(
                    my_pos,
                    best_patrol,
                    default=9999.0,
                )
                value -= patrol_weight * float(patrol_dist)

            if (not is_pacman) and self._intrusion_targets:
                min_intrusion_dist = min(
                    self._maze_distance(my_pos, pos, default=9999.0)
                    for (pos, ttl) in self._intrusion_targets
                )
                value -= self._INTRUSION_WEIGHT * float(min_intrusion_dist)

        if endgame_attack and is_pacman and num_invaders == 0 and not ally_pacman:
            value += 10.0

        last_action = state.get_last_agent_action(self.agent_index)
        reverse_penalty = 0.0
        if last_action is not None:
            rev = pacai.core.action.get_reverse_direction(last_action)
            if rev is not None and action == rev:
                reverse_penalty = self._REVERSE_PENALTY

        center_camp_penalty = 0.0
        if self._board is not None:
            width = self._board.width
            mid_col = width // 2
            dist_to_mid = abs(my_pos.col - mid_col)
            if (
                not is_pacman
                and num_invaders == 0
                and dist_to_mid <= 1
                and turn_count >= 150
            ):
                center_camp_penalty = 15.0

        value -= stop_penalty
        value -= cross_penalty
        value -= reverse_penalty
        value -= self._STEP_PENALTY
        value -= center_camp_penalty
        return value
