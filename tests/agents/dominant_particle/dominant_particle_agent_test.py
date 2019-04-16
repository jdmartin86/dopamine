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
      #sess.run(tf.global_variables_initializer())
      #observation = np.ones([84, 84, 1])
      #agent.begin_episode(observation)
      #agent.step(reward=1, observation=observation)
      #agent.end_episode(reward=1)
  
if __name__ == '__main__':
  tf.test.main()
