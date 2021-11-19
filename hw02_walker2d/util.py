from gym import RewardWrapper


class WalkerRewardShape(RewardWrapper):
    def reward(self, reward):
        reward /= 1000
        return reward
