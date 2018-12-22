# ICML 2019 Release
"""The Dominating Quantile Network agent.

    Implements the Dominating Quantile Network from ICML 2019
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import math


from dopamine.agents.rainbow import rainbow_agent
import numpy as np
import tensorflow as tf

import gin.tf

slim = tf.contrib.slim


@gin.configurable
class DominatingQuantileAgent(rainbow_agent.RainbowAgent):
  """An extension of Rainbow to perform dominating quantile regression."""

  def __init__(self,
               sess,
               num_actions,
               ssd_lambda=1.0,
               num_samples=32,
               num_quantiles=32,
               wass_xi = 1.0,
               wass_marginal_weight = 1.0,
               double_dqn=False,
               summary_writer=None,
               summary_writing_frequency=500):
    """Initializes the agent and constructs the Graph.

    Most of this constructor's parameters are IQN-specific hyperparameters whose
    values are taken from Dabney et al. (2018).

    Args:
      sess: `tf.Session` object for running associated ops.
      num_actions: int, number of actions the agent can take at any state.
      ssd_lambda: float, SSD regularization param.
      num_samples: int, number of quantile samples for loss
        estimation.
      num_quantiles: int, number of quantiles for computing Q-values.
      double_dqn: boolean, whether to perform double DQN style learning
        as described in Van Hasselt et al.: https://arxiv.org/abs/1509.06461.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
    """
    self.ssd_lambda = ssd_lambda
    # num_samples = M in the paper
    self.num_samples = num_samples
    # num_quantiles
    self.num_quantiles = num_quantiles
    # entropic regularized 2-Wasserstein temperature param
    self.wass_xi = wass_xi
    # marginal regularization param (alpha and beta in the paper)
    self.wass_marginal_weight = 1.0

    # option to perform double dqn.
    self.double_dqn = double_dqn
    # benchmark CVaR values (uniform over [0,10]) 
    # TODO: initialized in a more informed manner
    self.benchmark_cvar = tf.lin_space(0.0,10.0,self.num_quantiles)

    super(DominatingQuantileAgent, self).__init__(
        sess=sess,
        num_actions=num_actions,
        summary_writer=summary_writer,
        summary_writing_frequency=summary_writing_frequency)

  def _get_network_type(self):
    """Returns the type of the outputs of the implicit quantile network.

    Returns:
      _network_type object defining the outputs of the network.
    """
    return collections.namedtuple(
        'dqn_network', ['quantile_values', 'quantiles'])

  def _network_template(self, state, num_quantiles):
    r"""Builds an Dominating Quantile ConvNet.

    Takes state and quantile as inputs and outputs state-action quantile values.

    Args:
      state: A `tf.placeholder` for the RL state.
      num_quantiles: int, number of quantile inputs.

    Returns:
      _network_type object containing quantile value outputs of the network.
    """

    # specify parameter initialization
    weights_initializer = slim.variance_scaling_initializer(
        factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

    # normalize inputs to be in [0,1]
    net = tf.cast(state, tf.float32)
    net = tf.div(net, 255.)

    # build the DQN with three convolutional layers
    net = slim.conv2d(net, 32, [8, 8], stride=4,
        weights_initializer=weights_initializer)
    net = slim.conv2d(net, 64, [4, 4], stride=2,
        weights_initializer=weights_initializer)
    net = slim.conv2d(net, 64, [3, 3], stride=1,
        weights_initializer=weights_initializer)
    net = slim.flatten(net) # flatten conv output to |batch|x|flat out|

    net_size = net.get_shape().as_list()[-1]
    net_tiled = tf.tile(net, [num_quantiles, 1])

    batch_size = net.get_shape().as_list()[0]
    quantiles_shape = [num_quantiles * batch_size, 1]
    noise = tf.random_uniform(
        quantiles_shape, minval=0, maxval=1, dtype=tf.float32)

    # Hadamard product.
    net = tf.multiply(net_tiled, noise)

    net = slim.fully_connected(net, 512, 
        weights_initializer=weights_initializer)
    quantile_values = slim.fully_connected(net, self.num_actions,
        activation_fn=None,
        weights_initializer=weights_initializer)

    return self._get_network_type()(quantile_values=quantile_values,
                                    quantiles=noise)

  def _build_networks(self):
    """Builds the IQN computations needed for acting and training.

    These are:
      self.online_convnet: For computing the current state's quantile values.
      self.target_convnet: For computing the next state's target quantile
        values.
      self._net_outputs: The actual quantile values.
      self._q_argmax: The action maximizing the current state's Q-values.
      self._replay_net_outputs: The replayed states' quantile values.
      self._replay_next_target_net_outputs: The replayed next states' target
        quantile values.
    """
    # Calling online_convnet will generate a new graph as defined in
    # self._get_network_template using whatever input is passed, but will always
    # share the same weights.
    self.online_convnet = tf.make_template('Online', self._network_template)

    # Our implementation uses the Target network as a reference to older 
    # weight estimates. This corresponds to mu_k in the paper.
    self.target_convnet = tf.make_template('Target', self._network_template)

    # Compute the Q-values which are used for action selection in the current
    # state.
    self._net_outputs = self.online_convnet(self.state_ph,
                                            self.num_quantiles)
    # Shape of self._net_outputs.quantile_values:
    # num_quantiles x num_actions.
    # e.g. if num_actions is 2, it might look something like this:
    # Vals for Quantile .2  Vals for Quantile .4  Vals for Quantile .6
    #    [[0.1, 0.5],         [0.15, -0.3],          [0.15, -0.2]]
    # Q-values = [(0.1 + 0.15 + 0.15)/3, (0.5 + 0.15 + -0.2)/3].
    self._q_values = tf.reduce_mean(self._net_outputs.quantile_values, axis=0)
    self._q_argmax = tf.argmax(self._q_values, axis=0)

    self._replay_net_outputs = self.online_convnet(self._replay.states,
                                                   self.num_quantiles)
    # Shape: (num_quantiles x batch_size) x num_actions.
    self._replay_net_quantile_values = self._replay_net_outputs.quantile_values
    self._replay_net_quantiles = self._replay_net_outputs.quantiles

    # Do the same for next states in the replay buffer.
    self._replay_net_target_outputs = self.online_convnet( # uses online net here
        self._replay.next_states, self.num_quantiles)
    # Shape: (num_quantiles x batch_size) x num_actions.
    self._replay_net_target_quantile_values = self._replay_net_target_outputs.quantile_values

    # Do the same for references in the replay buffer.
    self._replay_net_reference_outputs = self.target_convnet( 
        self._replay.states, self.num_quantiles)
    # Shape: (num_quantiles x batch_size) x num_actions.
    self._replay_net_reference_quantile_values = self._replay_net_reference_outputs.quantile_values

    # Compute Q-values which are used for action selection for the next states
    # in the replay buffer. Compute the argmax over the Q-values.
    # (no double DQN here)
    outputs_action = self.online_convnet(self._replay.next_states,
                                        self.num_quantiles)

    # Shape: (num_quantiles x batch_size) x num_actions.
    target_quantile_values_action = outputs_action.quantile_values
    # Shape: num_quantiles x batch_size x num_actions.
    target_quantile_values_action = tf.reshape(target_quantile_values_action,
                                               [self.num_quantiles,
                                                self._replay.batch_size,
                                                self.num_actions])
    # Shape: batch_size x num_actions.
    self._replay_net_target_q_values = tf.squeeze(tf.reduce_mean(
        target_quantile_values_action, axis=0))
    self._replay_next_qt_argmax = tf.argmax(
        self._replay_net_target_q_values, axis=1)

  def _build_target_quantile_values_op(self):
    """Build an op used as a target for return values at given quantiles.

    Returns:
      An op calculating the target quantile return.
    """
    batch_size = tf.shape(self._replay.rewards)[0]
    # Shape of rewards: (num_quantiles x batch_size) x 1.
    rewards = self._replay.rewards[:, None]
    rewards = tf.tile(rewards, [self.num_quantiles, 1])

    is_terminal_multiplier = 1. - tf.to_float(self._replay.terminals)
    # Incorporate terminal state to discount factor.
    # size of gamma_with_terminal: (num_quantiles x batch_size) x 1.
    gamma_with_terminal = self.cumulative_gamma * is_terminal_multiplier
    gamma_with_terminal = tf.tile(gamma_with_terminal[:, None],
                                  [self.num_quantiles, 1])

    # Get the indices of the maximium Q-value across the action dimension.
    # Shape of replay_next_qt_argmax: (num_quantiles x batch_size) x 1.
    replay_next_qt_argmax = tf.tile(
        self._replay_next_qt_argmax[:, None], [self.num_quantiles, 1])

    # Shape of batch_indices: (num_quantiles x batch_size) x 1.
    batch_indices = tf.cast(tf.range(
        self.num_quantiles * batch_size)[:, None], tf.int64)

    # Shape of batch_indexed_target_values:
    # (num_quantiles x batch_size) x 2.
    batch_indexed_target_values = tf.concat(
        [batch_indices, replay_next_qt_argmax], axis=1)

    # Shape of next_target_values: (num_quantiles x batch_size) x 1.
    target_quantile_values = tf.gather_nd(
        self._replay_net_target_quantile_values,
        batch_indexed_target_values)[:, None]

    return rewards + gamma_with_terminal * target_quantile_values

  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """
    batch_size = tf.shape(self._replay.rewards)[0]

    target_quantile_values = tf.stop_gradient(
        self._build_target_quantile_values_op())
    # Reshape to self.num_quantiles x batch_size x 1 since this is
    # the manner in which the target_quantile_values are tiled.
    target_quantile_values = tf.reshape(target_quantile_values,
                                        [self.num_quantiles,
                                         batch_size, 1])
    # Transpose dimensions so that the dimensionality is batch_size x
    # self.num_quantiles x 1 to prepare for computation of
    # Bellman errors.
    # Final shape of target_quantile_values:
    # batch_size x num_quantiles x 1.
    target_quantile_values = tf.transpose(target_quantile_values, [1, 0, 2])

    # Shape of indices: (num_quantiles x batch_size) x 1.
    # Expand dimension by one so that it can be used to index into all the
    # quantiles when using the tf.gather_nd function (see below).
    indices = tf.range(self.num_quantiles * batch_size)[:, None]

    # Expand the dimension by one so that it can be used to index into all the
    # quantiles when using the tf.gather_nd function (see below).
    reshaped_actions = self._replay.actions[:, None]
    reshaped_actions = tf.tile(reshaped_actions, [self.num_quantiles, 1])

    # Shape of reshaped_actions: (num_quantiles x batch_size) x 2.
    reshaped_actions = tf.concat([indices, reshaped_actions], axis=1)

    chosen_action_quantile_values = tf.gather_nd(
        self._replay_net_quantile_values, reshaped_actions)
    # Reshape to self.num_quantiles x batch_size x 1 since this is the manner
    # in which the quantile values are tiled.
    chosen_action_quantile_values = tf.reshape(chosen_action_quantile_values,
                                               [self.num_quantiles,
                                                batch_size, 1])
    # Transpose dimensions so that the dimensionality is batch_size x
    # self.num_tau_samples x 1 to prepare for computation of
    # Bellman errors.
    # Final shape of chosen_action_quantile_values:
    # batch_size x num_quantiles x 1.
    chosen_action_quantile_values = tf.transpose(
        chosen_action_quantile_values, [1, 0, 2])

    # target values are the average over quantiles for a given sample
    # target_values shape: batch_size x 1
    target_values = tf.reduce_mean(target_quantile_values, axis=1)
    chosen_values = tf.reduce_mean(chosen_action_quantile_values,axis=1)
    
    # Shape of bellman_erors and huber_loss:
    # batch_size x num_quantiles x 1.    
    bellman_errors = target_values[:,None,:] - chosen_action_quantile_values

    # Bellman's potential energy
    # Shape: batch_size x num_quantiles x 1
    bellman_potential_energy = 0.5 * bellman_errors ** 2

    # SSD potential energy
    # Sort the target quantile values for CVaR computation
    # Shape of target_quantiles_sorted: batch_size x num_quantiles x 1.
    target_quantiles_sorted = tf.contrib.framework.sort(target_quantile_values, axis=1)
    target_cvars = tf.cumsum(target_quantiles_sorted, axis=1)
    ssd_potential_energy = target_cvars - tf.stop_gradient(self.benchmark_cvar[None,:,None])
    ssd_potential_energy = tf.to_float(ssd_potential_energy > 0.0) * self.ssd_lambda * ssd_potential_energy

    # total energy loss
    # Shape of total_energy: batch_size x num_quantiles x 1
    total_energy = tf.reduce_mean(bellman_potential_energy + ssd_potential_energy, axis=1)

    # Entropic Wasserstein loss
    
    # Shape of reference_quantile_values:  batch_size x num_quantiles x 1. 
    # Reshape to self.num_quantiles x batch_size x 1 since this is
    # the manner in which the target_quantile_values are tiled.
    reference_quantile_values = tf.stop_gradient(self._replay_net_reference_quantile_values)
    reference_quantile_values = tf.reshape(reference_quantile_values,
                                        [self.num_quantiles,
                                         batch_size, 1])
    # Transpose dimensions so that the dimensionality is batch_size x
    # self.num_quantiles x 1 to prepare for computation of
    # Bellman errors.
    # Final shape of target_quantile_values:
    # batch_size x num_quantiles x 1.
    reference_quantile_values = tf.transpose(reference_quantile_values, [1, 0, 2])

    # Compute the pairwise probability using a sq-euclidean cost
    # Shapes: batch_size x num_quantiles x num_quantiles x 1.    
    pairwise_cost = chosen_action_quantile_values[:,:,None,:] - reference_quantile_values[:,None,:,:]
    pairwise_cost = pairwise_cost**2
    pairwise_prob = tf.exp(-pairwise_cost/self.wass_xi-1.0-self.wass_marginal_weight/self.wass_xi) 

    # Shape of entropy: batch_size x num_quantiles x num_quantiles x 1
    pairwise_entr = pairwise_prob * tf.log(pairwise_prob)

    # Compute constraints for marginal projections: each needs to be 
    # a uniform distribution. 
    # TODO: apply unique vector weights to each
    # Shape of marginals: batch_size x num_quantiles x 1
    marginal_i = self.wass_marginal_weight*(tf.reduce_sum(pairwise_prob,axis=2) - 1.0/self.num_quantiles)
    marginal_j = self.wass_marginal_weight*(tf.reduce_sum(pairwise_prob,axis=1) - 1.0/self.num_quantiles)

    # Take the Frobenius norm to compute W2 metric
    # Shape: batch_size x 1. 
    wass_2_entropic = pairwise_prob*pairwise_cost + self.wass_xi*pairwise_entr + marginal_i + marginal_j
    wass_2_entropic = tf.reduce_sum(wass_2_entropic,axis=1)
    wass_2_entropic = tf.reduce_sum(wass_2_entropic,axis=2)
    
    # total JKO loss
    # Shape: batch_size x 1.
    loss = wass_2_entropic + total_energy

    # TODO(kumasaurabh): Add prioritized replay functionality here.
    update_priorities_op = tf.no_op()
    with tf.control_dependencies([update_priorities_op]):
      if self.summary_writer is not None:
        with tf.variable_scope('Losses'):
          tf.summary.scalar('QuantileLoss', tf.reduce_mean(loss))
      return self.optimizer.minimize(tf.reduce_mean(loss)), tf.reduce_mean(loss)
