# Copyright April 2019.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Dominate Particle Network (DPN) agent.

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
class DominantParticleAgent(rainbow_agent.RainbowAgent):
  """An extension of Rainbow."""

  def __init__(self,
               sess,
               num_actions,
               num_particles = 64,
               num_target_samples = 32,
               seed_dim = 5,
               blur = 0.05,
               scaling = 0.5,
               tau = 0.1,
               double_dqn=False,
               summary_writer=None,
               summary_writing_frequency=500):
    """Initializes the agent and constructs the Graph.

    Args:
      sess: `tf.Session` object for running associated ops.
      num_actions: int, number of actions the agent can take at any state.
      
      
      double_dqn: boolean, whether to perform double DQN style learning
        as described in Van Hasselt et al.: https://arxiv.org/abs/1509.06461.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
    """
    # Number of particles supporting discrete measures
    self.num_particles = num_particles
    # Number of target dists draws for averaging loss over one transition
    self.num_target_samples = num_target_samples
    # Dimensionality of input noise vector
    self.seed_dim = seed_dim
    # Simulated annealing log step size
    self.blur = blur
    # Simulated annealing max log scale
    self.scaling = scaling    
    # Simulated annealing scales for entropic regularization
    self.scales = [ np.exp(e) for e in np.arange(1, np.log(self.blur), np.log(self.scaling)) ] + [self.blur]
    # Temporal stepsize for potential energy functional (2*h in paper)
    self.tau= tau
    # Particle weights are all equally-likely
    self.a_i = tf.tile([1.0/self.num_particles],[self.num_particles])[:,None]
    # Particle weights in the log domain
    self.loga_i = tf.math.log(self.a_i)
    self.logb_i = tf.math.log(self.a_i)
    # Option to perform double dqn.
    self.double_dqn = double_dqn

    super(DominantParticleAgent, self).__init__(
        sess=sess,
        num_actions=num_actions,
        summary_writer=summary_writer,
        summary_writing_frequency=summary_writing_frequency)

  def _get_network_type(self):
    """Returns the type of the outputs of the Dominate Particle Network.

    Returns:
      _network_type object defining the outputs of the network.
    """
    return collections.namedtuple('dpa_network', ['particle_locs'])

  def _network_template(self, state, num_draws = 1):
    r"""Builds a Dominate Particle Network.

    Takes state and seed as inputs and outputs particle vectors that support
    finite return distributions for every action.

    Args:
      state: A `tf.placeholder` for the RL state.
      num_draws: int for the number of draws of return dists 

    Returns:
      _network_type object containing particle outputs of the network.
    """

    weights_initializer = slim.variance_scaling_initializer(
        factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

    # Prepare image (state) input
    state_net = tf.cast(state, tf.float32)
    state_net = tf.math.divide(state_net, 255.)

    # Tile the batch dimension for each network draw 
    # (batch_size x num_draws, state_dims)
    batch_size = state_net.get_shape().as_list()[0]
    state_net = tf.tile(state_net,[num_draws,1,1,1])

    # Convolutional portion to extract image features
    state_net = slim.conv2d(
        state_net, 32, [8, 8], stride=4,
        weights_initializer=weights_initializer)
    state_net = slim.conv2d(
        state_net, 64, [4, 4], stride=2,
        weights_initializer=weights_initializer)
    state_net = slim.conv2d(
        state_net, 64, [3, 3], stride=1,
        weights_initializer=weights_initializer)

    # Flatten image features and tile for num particles
    # (batch_size, feature_dim)
    state_net = slim.flatten(state_net) # image features

    # Prepare noise input
    # (batch_size x num_draws, seed_dim)
    seed_net = tf.random_normal([batch_size*num_draws,self.seed_dim],dtype=tf.float32)

    # Join the two image features and the seed as input to the
    # fully-connected portion of the network
    # (batch_size x num_draws x num_particles, feature_dim + seed_dim)
    net = tf.concat([state_net, seed_net],1) 
    net = tf.tile(net, [self.num_particles, 1])

    # The fully-connected portion maps image features and noise
    # to vectors of particle locations for each action
    net = slim.fully_connected(net, 512, weights_initializer=weights_initializer)
    particle_locs = slim.fully_connected(net,self.num_actions,
        activation_fn=None,weights_initializer=weights_initializer)

    return self._get_network_type()(particle_locs=particle_locs)

  def _build_networks(self):
    """Builds the DPN computations needed for acting and training.

    These are:
      self.online_convnet: For computing the current state's return particle locations.
      self.target_convnet: For computing the next state's target return particle locations
        values.
      self._net_outputs: The return particle locations.
      self._q_argmax: The action maximizing the current state's Q-values.
      self._replay_net_outputs: The replayed states' return particle locations.
      self._replay_next_target_net_outputs: The replayed next states' target return particle locations.
    """
    # Calling online_convnet will generate a new graph as defined in
    # self._get_network_template using whatever input is passed, but will always
    # share the same weights.
    self.online_convnet = tf.make_template('Online', self._network_template)
    self.target_convnet = tf.make_template('Target', self._network_template)

    # Compute the Q-values which are used for action selection in the current
    # state.
    self._net_outputs = self.online_convnet(self.state_ph,1)

    # Shape of self._net_outputs.quantile_values:
    # num_particles x num_actions.
    # e.g. if num_actions is 2, it might look something like this:
    # Vals for Quantile .2  Vals for Quantile .4  Vals for Quantile .6
    #    [[0.1, 0.5],         [0.15, -0.3],          [0.15, -0.2]]
    # Q-values = [(0.1 + 0.15 + 0.15)/3, (0.5 + 0.15 + -0.2)/3].
    self._q_values = tf.reduce_mean(self._net_outputs.particle_locs, axis=0)
    self._q_argmax = tf.argmax(self._q_values, axis=0)

    self._replay_net_outputs = self.online_convnet(self._replay.states, 1)
    # Shape: (batch_size x num_particles) x num_actions. 
    self._replay_net_particle_locs = self._replay_net_outputs.particle_locs

    # Do the same for next states in the replay buffer.
    self._replay_net_target_outputs = self.target_convnet(self._replay.next_states, self.num_target_samples)
    # Shape: (batch_size x num_target_samples x num_particles) x num_actions.
    vals = self._replay_net_target_outputs.particle_locs
    self._replay_net_target_particle_locs = vals

    # Compute Q-values which are used for action selection for the next states
    # in the replay buffer. Compute the argmax over the Q-values.
    if self.double_dqn:
      outputs_action = self.online_convnet(self._replay.next_states, self.num_target_samples)
    else:
      outputs_action = self.target_convnet(self._replay.next_states, self.num_target_samples)

    # Shape: (num_particles x batch_size x num_target_samples) x num_actions.
    target_particle_locs_action = outputs_action.particle_locs
    # Shape: num_particles x (batch_size x num_target_samples) x num_actions.
    target_particle_locs_action = tf.reshape(target_particle_locs_action,
                                               [self.num_particles,
                                                self._replay.batch_size*self.num_target_samples,
                                                self.num_actions])
    
    # Shape: (batch_size x num_target_samples) x num_actions.
    self._replay_net_target_q_values = tf.squeeze(tf.reduce_mean(target_particle_locs_action, axis=0))

    # Shape: (batch_size x num_target_samples) x 1
    self._replay_next_qt_argmax = tf.argmax(self._replay_net_target_q_values, axis=1)

  def _build_target_quantile_values_op(self):
    """Build an op used as a target for return values at given quantiles.

    Returns:
      An op calculating the target quantile return.
    """
    batch_size = tf.shape(self._replay.rewards)[0]
    # Shape of rewards: (num_target_samples x num_particles x batch_size) x 1.
    rewards = self._replay.rewards[:, None]
    rewards = tf.tile(rewards, [self.num_particles*self.num_target_samples, 1])

    # Incorporate terminal state to discount factor.
    # size of gamma_with_terminal: (num_target_samples x num_particles x batch_size) x 1.
    is_terminal_multiplier = 1. - tf.cast(self._replay.terminals,tf.float32)
    gamma_with_terminal = self.cumulative_gamma * is_terminal_multiplier
    gamma_with_terminal = tf.tile(gamma_with_terminal[:, None], [self.num_particles*self.num_target_samples, 1])

    # Get the indices of the maximium Q-value across the action dimension.
    # Shape of replay_next_qt_argmax: (batch_size x num_target_samples x num_particles) x 1.
    replay_next_qt_argmax = tf.tile(self._replay_next_qt_argmax[:, None], [self.num_particles, 1])

    # Shape of batch_indices: (batch_size x num_target_samples x num_particles) x 1.
    batch_indices = tf.cast(tf.range(batch_size * self.num_target_samples * self.num_particles)[:, None], tf.int64)

    # Shape of batch_indexed_target_values:
    # (batch_size x num_target_samples x num_particles) x 2.
    batch_indexed_target_values = tf.concat([batch_indices, replay_next_qt_argmax], axis=1)

    # Shape of next_target_values: (batch_size x num_target_samples x num_particles) x 1.
    target_particle_set = tf.gather_nd(
        self._replay_net_target_particle_locs,
        batch_indexed_target_values)[:, None]

    return rewards + gamma_with_terminal * target_particle_set

  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """
    batch_size = tf.shape(self._replay.rewards)[0]

    # Shape: (batch_size x num_target_samples x num_particles) x 1
    target_particle_set = tf.stop_gradient(self._build_target_quantile_values_op())

    # Reshape to num_particles x batch_size x 1 since this is
    # the manner in which the target_quantile_values are tiled.
    target_particle_set = tf.reshape(target_particle_set,
                                    [self.num_particles,
                                    batch_size,
                                    self.num_target_samples, 1])

    # Transpose dimensions so that the dimensionality is batch_size x
    # self.num_particles x 1 to prepare for computation of
    # Bellman errors.
    # Final shape of target_quantile_values:
    # batch_size x num_target_samples x num_particles x 1.
    target_particle_set = tf.transpose(target_particle_set, [1, 2, 0, 3])

    # Shape of indices: (batch_size x num_particles) x 1.
    # Expand dimension by one so that it can be used to index into all the
    # particles when using the tf.gather_nd function (see below).
    indices = tf.range(batch_size * self.num_particles)[:, None]

    # Expand the dimension by one so that it can be used to index into all the
    # quantiles when using the tf.gather_nd function (see below).
    reshaped_actions = self._replay.actions[:, None]
    reshaped_actions = tf.tile(reshaped_actions, [self.num_particles, 1])

    # Shape of reshaped_actions: (batch_size x num_particles) x 2.
    reshaped_actions = tf.concat([indices, reshaped_actions], axis=1)

    # Reshape to self.num_particles x batch_size x 1 since this is the manner
    # in which the quantile values are tiled.
    # (batch_size x num_target_samples) x num_particles x 1.
    chosen_action_particles = tf.gather_nd(self._replay_net_particle_locs, reshaped_actions)[:,None]
    chosen_action_particles = tf.tile(chosen_action_particles,[self.num_target_samples,1])
    chosen_action_particles = tf.reshape(chosen_action_particles, 
      [self.num_particles,batch_size,self.num_target_samples, 1])

    # Transpose dimensions so that the dimensionality is batch_size x
    # self.num_particles x 1 to prepare for computation of
    # Bellman errors.
    # Final shape of chosen_action_particles:
    # num_tau_samples = num_particles  (keep until code runs)
    # num_tau_prime_samples = num_of_target_samples (keep until code runs)
    # batch_size x num_particles x 1.
    chosen_action_particles = tf.transpose(chosen_action_particles, [1, 2, 0, 3])

    # Shape of bellman_erors and huber_loss:
    # batch_size x num_target_samples x num_particles x 1.
    bellman_errors = target_particle_set - chosen_action_particles

    # Compute the L2 loss over the particles
    # batch_size x num_target_samples x 1
    l2_loss = tf.reduce_mean(0.5 * bellman_errors**2 , axis=2)

    # Compute the entropy-regulated Wasserstein loss
    # e-scaling heuristic (aka. simulated annealing): 
    def KP_log(x,y,b_j_log, blur = 1.):
        """
        Kernel product in the log domain
        x: source for grad (batch,num_target_samples,num_particles,1)
        y: target with no grad (batch,num_target_samples,num_particles,1)
        """
        xmy = x[:,:,:,None,:] - y[:,:,None,:,:] # batch,samples,parts,parts,1, xmy[i,j,k] = (x_i[k]-y_j[k])
        C = - xmy**2 / (2*(blur**2))
        return (blur**2)*tf.reduce_logsumexp(C + b_j_log, axis=2,keepdims=True)

    # Solve the OT_e(a,b) problem
    f_i = tf.zeros([batch_size,self.num_target_samples,self.num_particles,1,1],dtype=tf.float32)    
    g_j = tf.zeros([batch_size,self.num_target_samples,self.num_particles,1,1],dtype=tf.float32)
    for scale in self.scales:
        g_j = -KP_log(target_particle_set, chosen_action_particles, f_i/scale**2 + self.loga_i, blur=scale)
        f_i = -KP_log(chosen_action_particles, target_particle_set, g_j/scale**2 + self.logb_i, blur=scale)
    
    # Return the dual cost OT_e(a,b), assuming convergence in the Sinkhorn loop
    # batch_size x num_target_samples x 1
    Wb_loss = tf.reduce_mean(tf.squeeze(f_i),axis=2) + tf.reduce_mean(tf.squeeze(g_j),axis=2) 

    # average loss over target samples
    # batch_size x 1
    proximal_loss_ij = tf.reduce_mean(Wb_loss + self.tau*l2_loss,axis=1)

    # total loss over replay buffer
    proximal_loss = tf.reduce_sum(proximal_loss_ij,axis=0)

    # TODO(kumasaurabh): Add prioritized replay functionality here.
    update_priorities_op = tf.no_op()
    with tf.control_dependencies([update_priorities_op]):
      if self.summary_writer is not None:
        with tf.variable_scope('Losses'):
          tf.summary.scalar('ProximalLoss', proximal_loss)
      return self.optimizer.minimize(proximal_loss), proximal_loss
