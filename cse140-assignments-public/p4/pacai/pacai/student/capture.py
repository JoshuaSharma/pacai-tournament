import random
import pacai.core.agent
import pacai.core.agentinfo
import pacai.core.gamestate
import pacai.core.action
import pacai.search.distance
import pacai.util.alias


def create_team() -> list[pacai.core.agentinfo.AgentInfo]:
    """
    Get the agent information that will be used to create a capture team.
    """
    agent1_info = pacai.core.agentinfo.AgentInfo(
        name=f"{__name__}.OffensiveSmartAgent")
    agent2_info = pacai.core.agentinfo.AgentInfo(
        name=f"{__name__}.DefensiveSmartAgent")

    return [agent1_info, agent2_info]


class BaseSmartAgent(pacai.core.agent.Agent):
    """
    A base agent with shared utilities for safe distance calculation and game analysis.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.distance_calculator = None

    def game_start(self, initial_state: pacai.core.gamestate.GameState) -> None:
        """Initialize the distance calculator."""
        self.distance_calculator = pacai.search.distance.DistancePreComputer()
        self.distance_calculator.compute(initial_state.board)

    def get_safe_maze_distance(self, pos1, pos2, state):
        """
        Safely calculate maze distance. Returns 999999 if positions are None (dead agent).
        """
        if pos1 is None or pos2 is None:
            return 999999

        if self.distance_calculator:
            # DistancePreComputer can return None if no path exists
            dist = self.distance_calculator.get_distance(pos1, pos2)
            if dist is not None:
                return dist

        # Fallback to Manhattan if maze distance is unavailable
        return pacai.search.distance.manhattan_distance(pos1, pos2, state)

    def get_action(self, state: pacai.core.gamestate.GameState) -> pacai.core.action.Action:
        """
        Common action selection loop: evaluate all legal actions and pick the best.
        """
        legal_actions = state.get_legal_actions()
        if not legal_actions:
            return pacai.core.action.STOP

        action_scores = []
        for action in legal_actions:
            # Skip stopping unless it's the only choice (prevent idling)
            if action == pacai.core.action.STOP and len(legal_actions) > 1:
                continue

            successor = state.generate_successor(action)
            score = self.evaluate_state(successor)
            action_scores.append((score, action))

        if not action_scores:
            return pacai.core.action.STOP

        # Greedily choose the best action, break ties randomly
        best_score = max(action_scores, key=lambda x: x[0])[0]
        best_actions = [a for s, a in action_scores if s == best_score]
        return self.rng.choice(best_actions)

    def evaluate_state(self, state: pacai.core.gamestate.GameState) -> float:
        raise NotImplementedError("Subclasses must implement evaluate_state")


class OffensiveSmartAgent(BaseSmartAgent):
    """
    An offensive agent that focuses on food but plays safely when winning.
    """

    def evaluate_state(self, state: pacai.core.gamestate.GameState) -> float:
        my_pos = state.get_agent_position(self.agent_index)

        # 1. Death Check: If agent died (pos is None), return massive penalty.
        if my_pos is None:
            return -999999.0

        score = 0.0

        # 2. Base Score: Reward having a high game score.
        # get_normalized_score returns higher numbers for better results for OUR team.
        current_score = state.get_normalized_score(self.agent_index)
        score += current_score * 100

        # 3. Dynamic Mode Switching:
        # If we are winning by a lot (> 50 points), play defensively to burn time.
        # This secures the win and prevents risky deaths.
        if current_score > 50:
            # Defensive behavior: Stay on home side, near food
            if state.is_pacman(self.agent_index):
                score -= 1000  # Return home immediately
            else:
                # Patrol border/defend
                invaders = state.get_invader_positions(self.agent_index)
                if invaders:
                    dists = [self.get_safe_maze_distance(
                        my_pos, pos, state) for pos in invaders.values()]
                    score -= min(dists) * 10
                else:
                    # Stay near center border
                    center_pos = pacai.core.board.Position(
                        state.board.height // 2, state.board.width // 2)
                    score -= self.get_safe_maze_distance(
                        my_pos, center_pos, state)
            return score

        # --- Standard Offensive Logic ---

        # 4. Food Acquisition
        food_positions = state.get_food(agent_index=self.agent_index)
        if food_positions:
            # Find closest food
            min_food_dist = min([self.get_safe_maze_distance(
                my_pos, food, state) for food in food_positions])
            score -= min_food_dist * 2

        # 6. Ghost Avoidance (Survival)
        # This is critical. High penalty for being near non-scared ghosts.
        opponents = state.get_nonscared_opponent_positions(self.agent_index)
        if opponents and state.is_pacman(self.agent_index):
            dists = [self.get_safe_maze_distance(
                my_pos, pos, state) for pos in opponents.values()]
            min_ghost_dist = min(dists) if dists else 999

            if min_ghost_dist <= 1:
                score -= 10000  # Dead
            elif min_ghost_dist <= 2:
                score -= 5000   # Critical danger
            elif min_ghost_dist <= 4:
                score -= 500    # Warning zone

        # 7. Hunting Scared Ghosts
        scared_opponents = state.get_scared_opponent_positions(
            self.agent_index)
        if scared_opponents and state.is_pacman(self.agent_index):
            dists = [self.get_safe_maze_distance(
                my_pos, pos, state) for pos in scared_opponents.values()]
            min_scared_dist = min(dists) if dists else 999
            # Reward eating scared ghosts
            score -= min_scared_dist * 10

        return score


class DefensiveSmartAgent(BaseSmartAgent):
    """
    A defensive agent that intercepts invaders by predicting their targets.
    """

    def evaluate_state(self, state: pacai.core.gamestate.GameState) -> float:
        my_pos = state.get_agent_position(self.agent_index)

        if my_pos is None:
            return -999999.0

        score = 0.0

        # 1. Invader Handling
        invaders = state.get_invader_positions(self.agent_index)
        if invaders:
            # Calculate distance to the closest invader
            dists = [self.get_safe_maze_distance(
                my_pos, pos, state) for pos in invaders.values()]
            min_invader_dist = min(dists) if dists else 999

            if state.is_scared(self.agent_index):
                # If we are scared, run away!
                score += min_invader_dist * 20
                if min_invader_dist <= 1:
                    score -= 10000
            else:
                # If we are strong, chase!
                score -= min_invader_dist * 10
                # Bonus for catching
                if min_invader_dist <= 1:
                    score += 10000

                # INTERCEPTION LOGIC:
                # Instead of just chasing the invader's current position,
                # defend the food they are likely running towards.
                my_food = state.get_food(
                    team_modifier=0, agent_index=self.agent_index)
                if my_food:
                    # Find food closest to the invader
                    invader_pos = list(invaders.values())[
                        0]  # Simplify to first invader
                    closest_food_to_invader = min(
                        my_food, key=lambda f: self.get_safe_maze_distance(invader_pos, f, state))

                    # Distance from ME to that Threatened Food
                    dist_to_threat = self.get_safe_maze_distance(
                        my_pos, closest_food_to_invader, state)
                    # We want to be close to the threatened food to block them
                    score -= dist_to_threat * 5

        # 2. Patrol / Positioning (No Invaders)
        else:
            # If no invaders, stay on our side, preferably near the "border" or middle of our food
            # This minimizes the time to react to new invaders.
            if state.is_pacman(self.agent_index):
                score -= 1000  # Get back to defense!

            # Patrol center of board height (choke point)
            mid_height = state.board.height // 2
            mid_width = state.board.width // 2
            # Adjust width to be on our side (approximate)
            home_x = int(
                mid_width - 2) if self.agent_index % 2 == 0 else int(mid_width + 2)

            patrol_target = pacai.core.board.Position(mid_height, home_x)
            score -= self.get_safe_maze_distance(my_pos,
                                                 patrol_target, state) * 0.5

        return score