from src.sim.rewards.base_reward import BaseReward

class ConstantReward(BaseReward):
    """
    Reward that gives a constant value to all participants.
    """
    def calculate(self, world, before, participants):
        return {
            agent_id: 1.0
            for agent_id in participants
        }