from .pg import Policy


class ImitationLearner(Policy):
    """
        Actor-Critic agent which supports imitated learning scheme
    """

    def imitation_learning(self, inputs, rnn_hxs, masks, expert):
        """
            Imitation learning loss
        :param inputs: state observations
        :param rnn_hxs: rnn hidden state
        :param masks: mask the final state with 0 value
        :param expert: a trained or heuristic agent
        :return: log probability of expert's actions
        """
        _, actor_features, _ = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        expert_actions = expert.act(inputs)
        action_log_probs = dist.log_probs(expert_actions)

        return action_log_probs
