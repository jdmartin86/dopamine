# Copyright 2018 The Dopamine Authors.
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
"""Tests for the Dominant Particle Agent.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from dopamine.agents.dqn import dqn_agent
from dopamine.agents.dominant_particle import dominant_particle_agent
import numpy as np
import tensorflow as tf

slim = tf.contrib.slim


class DominantParticleAgentTest(tf.test.TestCase):

  def setUp(self):
    self._num_actions = 4
    self.observation_shape = dqn_agent.OBSERVATION_SHAPE
    self.stack_size = dqn_agent.STACK_SIZE
    self.ones_state = np.ones(
        [1, self.observation_shape, self.observation_shape, self.stack_size])
  
  def testCreateAgentWithDefaults(self):
    # Verifies that we can create and train an agent with the default values.
    with self.test_session(use_gpu=False) as sess:
      agent = dominant_particle_agent.DominantParticleAgent(sess, num_actions=4)
      sess.run(tf.global_variables_initializer())
      observation = np.ones([84, 84, 1])
      agent.begin_episode(observation)
      agent.step(reward=1, observation=observation)
      agent.end_episode(reward=1)
  
  def testShapes(self):
    with self.test_session(use_gpu=False) as sess:
      agent = dominant_particle_agent.DominantParticleAgent(sess, num_actions=4)

      # Replay buffer batch size:
      self.assertEqual(agent._replay.batch_size, 32)

      # particle locs, q-values, q-argmax at sample action time:
      self.assertEqual(agent._net_outputs.particle_locs.shape[0],
                       agent.num_particles)
      self.assertEqual(agent._net_outputs.particle_locs.shape[1],
                       agent.num_actions)
      self.assertEqual(agent._q_values.shape[0], agent.num_actions)

      # Check the setting of num_actions.
      self.assertEqual(self._num_actions, agent.num_actions)

      # input particles, particle locs, and output q-values at loss
      # computation time.
      self.assertEqual(agent._replay_net_particle_locs.shape[0],
                       agent.num_particles * agent._replay.batch_size)
      self.assertEqual(agent._replay_net_particle_locs.shape[1],
                       agent.num_actions)

      self.assertEqual(agent._replay_net_target_particle_locs.shape[0],
                       agent.num_target_samples * agent._replay.batch_size * agent.num_particles)
      self.assertEqual(agent._replay_net_target_particle_locs.shape[1],
                       agent.num_actions)

      self.assertEqual(agent._replay_net_target_q_values.shape[0],
                       agent.num_target_samples * agent._replay.batch_size)
      self.assertEqual(agent._replay_net_target_q_values.shape[1],
                       agent.num_actions)

  def test_q_value_computation(self):
    with self.test_session(use_gpu=False) as sess:
      agent = dominant_particle_agent.DominantParticleAgent(sess, num_actions=4)
      agent.eval_mode = True
      sess.run(tf.global_variables_initializer())

      state = self.ones_state
      feed_dict = {agent.state_ph: state}
      q_values, q_argmax = sess.run([agent._q_values, agent._q_argmax],
                                    feed_dict)
      for i in range(agent.num_actions):
        print("q_values: ", q_values[i])
      print("q_argmax: ", q_argmax)
      
if __name__ == '__main__':
  tf.test.main()
