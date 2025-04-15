
from pathlib import Path
from typing import List

from .types import ParsedTrajectory


class FileParser:

    def _parse_single_trajectory(self, lines: List[str]) -> ParsedTrajectory:
        """Parse a single trajectory from a list of lines.

        Returns:
            ParsedTrajectory with states, actions, and length.
        """
        states = []
        actions = []
        for line in lines:
            state_str, action = line.split()
            state = [int(x) for x in state_str.split(',')]
            action_idx = {'U': 0, 'R': 1, 'D': 2, 'L': 3}[action]
            states.append(state)
            actions.append(action_idx)
        return ParsedTrajectory(states=states, actions=actions, length=len(states))

    def parse_game_log(self, file_path: Path) -> List[ParsedTrajectory]:
        """Parse a single game log file into a list of trajectories.

        Args:
            file_path: Path to the game log file.

        Returns:
            List of ParsedTrajectory objects.
        """
        trajectories = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            current_trajectory: List[str] = []
            for line in lines:
                if line == 'NEW GAME\n':
                    if current_trajectory:
                        trajectories.append(self._parse_single_trajectory(current_trajectory))
                    current_trajectory = []
                else:
                    current_trajectory.append(line.strip())
            if current_trajectory:
                trajectories.append(self._parse_single_trajectory(current_trajectory))
        return trajectories