import os
import sys
import tensorflow as tf
import numpy as np
import scipy.stats as ss
from collections import deque
import cv2
import imageio

from params import train_params, test_params, play_params
from utils.network import Actor, Actor_BN
from utils.env_wrapper import PendulumWrapper, LunarLanderContinuousWrapper, BipedalWalkerWrapper

class Agent:
  
    def __init__(self, sess, env, seed, n_agent=0):
        print("Initialising agent %02d... \n" % n_agent)
         
        self.sess = sess        
        self.n_agent = n_agent
       
        # Create environment    
        if env == 'Pendulum-v0':
            self.env_wrapper = PendulumWrapper()
        elif env == 'LunarLanderContinuous-v2':
            self.env_wrapper = LunarLanderContinuousWrapper()
        elif env == 'BipedalWalker-v2':
            self.env_wrapper = BipedalWalkerWrapper()
        elif env == 'BipedalWalkerHardcore-v2':
            self.env_wrapper = BipedalWalkerWrapper(hardcore=True)
        else:
            raise Exception('Chosen environment does not have an environment wrapper defined. Please choose an environment with an environment wrapper defined, or create a wrapper for this environment in utils.env_wrapper.py')
        self.env_wrapper.set_random_seed(seed*(n_agent+1))
              
    def build_network(self, training):
        # Input placeholder    
        self.state_ph = tf.placeholder(tf.float32, ((None,) + train_params.STATE_DIMS)) 
        
        if training:
            # each agent has their own var_scope
            var_scope = ('actor_agent_%02d'%self.n_agent)
        else:
            # when testing, var_scope comes from main learner policy (actor) network
            var_scope = ('learner_actor_main')
          
        # Create policy (actor) network
        if train_params.USE_BATCH_NORM:
            self.actor_net = Actor_BN(self.state_ph, train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.ACTION_BOUND_LOW, train_params.ACTION_BOUND_HIGH, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, is_training=False, scope=var_scope)
            self.agent_policy_params = self.actor_net.network_params + self.actor_net.bn_params
        else:
            self.actor_net = Actor(self.state_ph, train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.ACTION_BOUND_LOW, train_params.ACTION_BOUND_HIGH, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, scope=var_scope)
            self.agent_policy_params = self.actor_net.network_params
                        
    def build_update_op(self, learner_policy_params):
        # Update agent's policy network params from learner
        update_op = []
        from_vars = learner_policy_params
        to_vars = self.agent_policy_params
                
        for from_var,to_var in zip(from_vars,to_vars):
            update_op.append(to_var.assign(from_var))
        
        self.update_op = update_op
                        
    def build_summaries(self, logdir):
        # Create summary writer to write summaries to disk
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        self.summary_writer = tf.summary.FileWriter(logdir, self.sess.graph)
        
        # Create summary op to save episode reward to Tensorboard log
        self.ep_reward_var = tf.Variable(0.0, trainable=False, name=('ep_reward_agent_%02d'%self.n_agent))
        tf.summary.scalar("Episode Reward", self.ep_reward_var)
        self.summary_op = tf.summary.merge_all()
        
        # Initialise reward var - this will not be initialised with the other network variables as these are copied over from the learner
        self.init_reward_var = tf.variables_initializer([self.ep_reward_var])
            
    def run(self, PER_memory, gaussian_noise, run_agent_event, stop_agent_event):
        # Continuously run agent in environment to collect experiences and add to replay memory
                
        # Initialise deque buffer to store experiences for N-step returns
        self.exp_buffer = deque()
        
        # Perform initial copy of params from learner to agent
        self.sess.run(self.update_op)
        
        # Initialise var for logging episode reward
        if train_params.LOG_DIR is not None:
            self.sess.run(self.init_reward_var)
        
        # Initially set threading event to allow agent to run until told otherwise
        run_agent_event.set()
        
        num_eps = 0
        
        while not stop_agent_event.is_set():
            num_eps += 1
            # Reset environment and experience buffer
            state = self.env_wrapper.reset()
            state = self.env_wrapper.normalise_state(state)
            self.exp_buffer.clear()
            
            num_steps = 0
            episode_reward = 0
            ep_done = False
            
            while not ep_done:
                num_steps += 1
                ## Take action and store experience
                if train_params.RENDER:
                    self.env_wrapper.render()
                action = self.sess.run(self.actor_net.output, {self.state_ph:np.expand_dims(state, 0)})[0]     # Add batch dimension to single state input, and remove batch dimension from single action output
                action += (gaussian_noise() * train_params.NOISE_DECAY**num_eps)
                next_state, reward, terminal = self.env_wrapper.step(action)
                
                episode_reward += reward 
                               
                next_state = self.env_wrapper.normalise_state(next_state)
                reward = self.env_wrapper.normalise_reward(reward)
                
                self.exp_buffer.append((state, action, reward))
                
                # We need at least N steps in the experience buffer before we can compute Bellman rewards and add an N-step experience to replay memory
                if len(self.exp_buffer) >= train_params.N_STEP_RETURNS:
                    state_0, action_0, reward_0 = self.exp_buffer.popleft()
                    discounted_reward = reward_0
                    gamma = train_params.DISCOUNT_RATE
                    for (_, _, r_i) in self.exp_buffer:
                        discounted_reward += r_i * gamma
                        gamma *= train_params.DISCOUNT_RATE
                    
                    # If learner is requesting a pause (to remove samples from PER), wait before adding more samples
                    run_agent_event.wait()   
                    PER_memory.add(state_0, action_0, discounted_reward, next_state, terminal, gamma)
                
                state = next_state
                
                if terminal or num_steps == train_params.MAX_EP_LENGTH:
                    # Log total episode reward
                    if train_params.LOG_DIR is not None:
                        summary_str = self.sess.run(self.summary_op, {self.ep_reward_var: episode_reward})
                        self.summary_writer.add_summary(summary_str, num_eps)
                    # Compute Bellman rewards and add experiences to replay memory for the last N-1 experiences still remaining in the experience buffer
                    while len(self.exp_buffer) != 0:
                        state_0, action_0, reward_0 = self.exp_buffer.popleft()
                        discounted_reward = reward_0
                        gamma = train_params.DISCOUNT_RATE
                        for (_, _, r_i) in self.exp_buffer:
                            discounted_reward += r_i * gamma
                            gamma *= train_params.DISCOUNT_RATE
                        
                        # If learner is requesting a pause (to remove samples from PER), wait before adding more samples
                        run_agent_event.wait()     
                        PER_memory.add(state_0, action_0, discounted_reward, next_state, terminal, gamma)
                    
                    # Start next episode
                    ep_done = True
                
            # Update agent networks with learner params every 'update_agent_ep' episodes
            if num_eps % train_params.UPDATE_AGENT_EP == 0:
                self.sess.run(self.update_op)
        
        self.env_wrapper.close()
    
    def test(self):   
        # Test a saved ckpt of actor network and save results to file (optional)
        
        def load_ckpt(ckpt_dir, ckpt_file):
            # Load ckpt given by ckpt_file, or else load latest ckpt in ckpt_dir
            loader = tf.train.Saver()    
            if ckpt_file is not None:
                ckpt = ckpt_dir + '/' + ckpt_file  
            else:
                ckpt = tf.train.latest_checkpoint(ckpt_dir)
             
            loader.restore(self.sess, ckpt)
            sys.stdout.write('%s restored.\n\n' % ckpt)
            sys.stdout.flush() 
             
            ckpt_split = ckpt.split('-')
            self.train_ep = ckpt_split[-1]
        
        # Load ckpt from ckpt_dir
        load_ckpt(test_params.CKPT_DIR, test_params.CKPT_FILE)
        
        # Create Tensorboard summaries to save episode rewards
        if test_params.LOG_DIR is not None:
            self.build_summaries(test_params.LOG_DIR)
            
        rewards = [] 

        for test_ep in range(1, test_params.NUM_EPS_TEST+1):
            state = self.env_wrapper.reset()
            state = self.env_wrapper.normalise_state(state)
            ep_reward = 0
            step = 0
            ep_done = False
            
            while not ep_done:
                if test_params.RENDER:
                    self.env_wrapper.render()
                action = self.sess.run(self.actor_net.output, {self.state_ph:np.expand_dims(state, 0)})[0]     # Add batch dimension to single state input, and remove batch dimension from single action output
                state, reward, terminal = self.env_wrapper.step(action)
                state = self.env_wrapper.normalise_state(state)
                
                ep_reward += reward
                step += 1
                 
                # Episode can finish either by reaching terminal state or max episode steps
                if terminal or step == test_params.MAX_EP_LENGTH:
                    sys.stdout.write('\x1b[2K\rTest episode {:d}/{:d}'.format(test_ep, test_params.NUM_EPS_TEST))
                    sys.stdout.flush()   
                    rewards.append(ep_reward)
                    ep_done = True   
                
        mean_reward = np.mean(rewards)
        error_reward = ss.sem(rewards)
                
        sys.stdout.write('\x1b[2K\rTesting complete \t Average reward = {:.2f} +/- {:.2f} /ep \n\n'.format(mean_reward, error_reward))
        sys.stdout.flush()  
        
        # Log average episode reward for Tensorboard visualisation
        if test_params.LOG_DIR is not None:
            summary_str = self.sess.run(self.summary_op, {self.ep_reward_var: mean_reward})
            self.summary_writer.add_summary(summary_str, self.train_ep)
         
        # Write results to file        
        if test_params.RESULTS_DIR is not None:
            if not os.path.exists(test_params.RESULTS_DIR):
                os.makedirs(test_params.RESULTS_DIR)
            output_file = open(test_params.RESULTS_DIR + '/' + test_params.ENV + '.txt' , 'a')
            output_file.write('Training Episode {}: \t Average reward = {:.2f} +/- {:.2f} /ep \n\n'.format(self.train_ep, mean_reward, error_reward))
            output_file.flush()
            sys.stdout.write('Results saved to file \n\n')
            sys.stdout.flush()      
        
        self.env_wrapper.close()       
        
    def play(self):   
        # Play a saved ckpt of actor network in the environment, visualise performance on screen and save a GIF (optional)
        
        def load_ckpt(ckpt_dir, ckpt_file):
            # Load ckpt given by ckpt_file, or else load latest ckpt in ckpt_dir
            loader = tf.train.Saver()    
            if ckpt_file is not None:
                ckpt = ckpt_dir + '/' + ckpt_file  
            else:
                ckpt = tf.train.latest_checkpoint(ckpt_dir)
        
            loader.restore(self.sess, ckpt)
            sys.stdout.write('%s restored.\n\n' % ckpt)
            sys.stdout.flush() 
             
            ckpt_split = ckpt.split('-')
            self.train_ep = ckpt_split[-1]
        
        # Load ckpt from ckpt_dir
        load_ckpt(play_params.CKPT_DIR, play_params.CKPT_FILE)
        
        # Create record directory
        if not os.path.exists(play_params.RECORD_DIR):
            os.makedirs(play_params.RECORD_DIR)

        for ep in range(1, play_params.NUM_EPS_PLAY+1):
            state = self.env_wrapper.reset()
            state = self.env_wrapper.normalise_state(state)
            step = 0
            ep_done = False
            
            while not ep_done:
                frame = self.env_wrapper.render()
                if play_params.RECORD_DIR is not None:
                    filepath = play_params.RECORD_DIR + '/Ep%03d_Step%04d.jpg' % (ep, step)
                    cv2.imwrite(filepath, frame)
                action = self.sess.run(self.actor_net.output, {self.state_ph:np.expand_dims(state, 0)})[0]     # Add batch dimension to single state input, and remove batch dimension from single action output
                state, _, terminal = self.env_wrapper.step(action)
                state = self.env_wrapper.normalise_state(state)
                
                step += 1
                 
                # Episode can finish either by reaching terminal state or max episode steps
                if terminal or step == play_params.MAX_EP_LENGTH:
                    ep_done = True   
                    
        # Convert saved frames to gif
        if play_params.RECORD_DIR is not None:
            images = []
            for file in sorted(os.listdir(play_params.RECORD_DIR)):
                # Load image
                filename = play_params.RECORD_DIR + '/' + file
                if filename.split('.')[-1] == 'jpg':
                    im = cv2.imread(filename)
                    images.append(im)
                    # Delete static image once loaded
                    os.remove(filename)
                 
            # Save as gif
            imageio.mimsave(play_params.RECORD_DIR + '/%s.gif' % play_params.ENV, images, duration=0.01)  
                    
        self.env_wrapper.close()     

class Learner:
    def __init__(self, sess, PER_memory, run_agent_event, stop_agent_event):
        print("Initialising learner... \n\n")
        
        self.sess = sess
        self.PER_memory = PER_memory
        self.run_agent_event = run_agent_event
        self.stop_agent_event = stop_agent_event
        
    def build_network(self):
        
        # Define input placeholders    
        self.state_ph = tf.placeholder(tf.float32, ((train_params.BATCH_SIZE,) + train_params.STATE_DIMS))
        self.action_ph = tf.placeholder(tf.float32, ((train_params.BATCH_SIZE,) + train_params.ACTION_DIMS))
        self.target_atoms_ph = tf.placeholder(tf.float32, (train_params.BATCH_SIZE, train_params.NUM_ATOMS)) # Atom values of target network with Bellman update applied
        self.target_Z_ph = tf.placeholder(tf.float32, (train_params.BATCH_SIZE, train_params.NUM_ATOMS))  # Future Z-distribution - for critic training
        self.action_grads_ph = tf.placeholder(tf.float32, ((train_params.BATCH_SIZE,) + train_params.ACTION_DIMS)) # Gradient of critic's value output wrt action input - for actor training
        self.weights_ph = tf.placeholder(tf.float32, (train_params.BATCH_SIZE)) # Batch of IS weights to weigh gradient updates based on sample priorities

        # Create value (critic) network + target network
        if train_params.USE_BATCH_NORM:
            self.critic_net = Critic_BN(self.state_ph, self.action_ph, train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, train_params.NUM_ATOMS, train_params.V_MIN, train_params.V_MAX, is_training=True, scope='learner_critic_main')
            self.critic_target_net = Critic_BN(self.state_ph, self.action_ph, train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, train_params.NUM_ATOMS, train_params.V_MIN, train_params.V_MAX, is_training=True, scope='learner_critic_target')
        else:
            self.critic_net = Critic(self.state_ph, self.action_ph, train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, train_params.NUM_ATOMS, train_params.V_MIN, train_params.V_MAX, scope='learner_critic_main')
            self.critic_target_net = Critic(self.state_ph, self.action_ph, train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, train_params.NUM_ATOMS, train_params.V_MIN, train_params.V_MAX, scope='learner_critic_target')
        
        # Create policy (actor) network + target network
        if train_params.USE_BATCH_NORM:
            self.actor_net = Actor_BN(self.state_ph, train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.ACTION_BOUND_LOW, train_params.ACTION_BOUND_HIGH, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, is_training=True, scope='learner_actor_main')
            self.actor_target_net = Actor_BN(self.state_ph, train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.ACTION_BOUND_LOW, train_params.ACTION_BOUND_HIGH, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, is_training=True, scope='learner_actor_target')
        else:
            self.actor_net = Actor(self.state_ph, train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.ACTION_BOUND_LOW, train_params.ACTION_BOUND_HIGH, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, scope='learner_actor_main')
            self.actor_target_net = Actor(self.state_ph, train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.ACTION_BOUND_LOW, train_params.ACTION_BOUND_HIGH, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, scope='learner_actor_target')
     
        # Create training step ops
        self.critic_train_step = self.critic_net.train_step(self.target_Z_ph, self.target_atoms_ph, self.weights_ph, train_params.CRITIC_LEARNING_RATE, train_params.CRITIC_L2_LAMBDA)
        self.actor_train_step = self.actor_net.train_step(self.action_grads_ph, train_params.ACTOR_LEARNING_RATE, train_params.BATCH_SIZE)
        
        # Create saver for saving model ckpts (we only save learner network vars)
        model_name = train_params.ENV + '.ckpt'
        self.checkpoint_path = os.path.join(train_params.CKPT_DIR, model_name)        
        if not os.path.exists(train_params.CKPT_DIR):
            os.makedirs(train_params.CKPT_DIR)
        saver_vars = [v for v in tf.global_variables() if 'learner' in v.name]
        self.saver = tf.train.Saver(var_list = saver_vars, max_to_keep=201) 
        
    def build_update_ops(self):     
        network_params = self.actor_net.network_params + self.critic_net.network_params
        target_network_params = self.actor_target_net.network_params + self.critic_target_net.network_params
        
        # Create ops which update target network params with hard copy of main network params
        init_update_op = []
        for from_var,to_var in zip(network_params, target_network_params):
            init_update_op.append(to_var.assign(from_var))
        
        # Create ops which update target network params with fraction of (tau) main network params
        update_op = []
        for from_var,to_var in zip(network_params, target_network_params):
            update_op.append(to_var.assign((tf.multiply(from_var, train_params.TAU) + tf.multiply(to_var, 1. - train_params.TAU))))        
            
        self.init_update_op = init_update_op
        self.update_op = update_op
    
    def initialise_vars(self):
        # Load ckpt file if given, otherwise initialise variables and hard copy to target networks
        if train_params.CKPT_FILE is not None:
            #Restore all learner variables from ckpt
            ckpt = train_params.CKPT_DIR + '/' + train_params.CKPT_FILE
            ckpt_split = ckpt.split('-')
            step_str = ckpt_split[-1]
            self.start_step = int(step_str)    
            self.saver.restore(self.sess, ckpt)
        else:
            self.start_step = 0
            self.sess.run(tf.global_variables_initializer())   
            # Perform hard copy (tau=1.0) of initial params to target networks
            self.sess.run(self.init_update_op)
            
    def run(self):
        # Sample batches of experiences from replay memory and train learner networks 
            
        # Initialise beta to start value
        priority_beta = train_params.PRIORITY_BETA_START
        beta_increment = (train_params.PRIORITY_BETA_END - train_params.PRIORITY_BETA_START) / train_params.NUM_STEPS_TRAIN
        
        # Can only train when we have at least batch_size num of samples in replay memory
        while len(self.PER_memory) <= train_params.BATCH_SIZE:
            sys.stdout.write('\rPopulating replay memory up to batch_size samples...')   
            sys.stdout.flush()
        
        # Training
        sys.stdout.write('\n\nTraining...\n')   
        sys.stdout.flush()
    
        for train_step in range(self.start_step+1, train_params.NUM_STEPS_TRAIN+1):  
            # Get minibatch
            minibatch = self.PER_memory.sample(train_params.BATCH_SIZE, priority_beta) 
            
            states_batch = minibatch[0]
            actions_batch = minibatch[1]
            rewards_batch = minibatch[2]
            next_states_batch = minibatch[3]
            terminals_batch = minibatch[4]
            gammas_batch = minibatch[5]
            weights_batch = minibatch[6]
            idx_batch = minibatch[7]            
    
            # Critic training step    
            # Predict actions for next states by passing next states through policy target network
            future_action = self.sess.run(self.actor_target_net.output, {self.state_ph:next_states_batch})  
            # Predict future Z distribution by passing next states and actions through value target network, also get target network's Z-atom values
            target_Z_dist, target_Z_atoms = self.sess.run([self.critic_target_net.output_probs, self.critic_target_net.z_atoms], {self.state_ph:next_states_batch, self.action_ph:future_action})
            # Create batch of target network's Z-atoms
            target_Z_atoms = np.repeat(np.expand_dims(target_Z_atoms, axis=0), train_params.BATCH_SIZE, axis=0)
            # Value of terminal states is 0 by definition
            target_Z_atoms[terminals_batch, :] = 0.0
            # Apply Bellman update to each atom
            target_Z_atoms = np.expand_dims(rewards_batch, axis=1) + (target_Z_atoms*np.expand_dims(gammas_batch, axis=1))
            # Train critic
            TD_error, _ = self.sess.run([self.critic_net.loss, self.critic_train_step], {self.state_ph:states_batch, self.action_ph:actions_batch, self.target_Z_ph:target_Z_dist, self.target_atoms_ph:target_Z_atoms, self.weights_ph:weights_batch})   
            # Use critic TD errors to update sample priorities
            self.PER_memory.update_priorities(idx_batch, (np.abs(TD_error)+train_params.PRIORITY_EPSILON))
                        
            # Actor training step
            # Get policy network's action outputs for selected states
            actor_actions = self.sess.run(self.actor_net.output, {self.state_ph:states_batch})
            # Compute gradients of critic's value output distribution wrt actions
            action_grads = self.sess.run(self.critic_net.action_grads, {self.state_ph:states_batch, self.action_ph:actor_actions})
            # Train actor
            self.sess.run(self.actor_train_step, {self.state_ph:states_batch, self.action_grads_ph:action_grads[0]})
            
            # Update target networks
            self.sess.run(self.update_op)
            
            # Increment beta value at end of every step   
            priority_beta += beta_increment
                            
            # Periodically check capacity of replay mem and remove samples (by FIFO process) above this capacity
            if train_step % train_params.REPLAY_MEM_REMOVE_STEP == 0:
                if len(self.PER_memory) > train_params.REPLAY_MEM_SIZE:
                    # Prevent agent from adding new experiences to replay memory while learner removes samples
                    self.run_agent_event.clear()
                    samples_to_remove = len(self.PER_memory) - train_params.REPLAY_MEM_SIZE
                    self.PER_memory.remove(samples_to_remove)
                    # Allow agent to continue adding experiences to replay memory
                    self.run_agent_event.set()
                    
            sys.stdout.write('\rStep {:d}/{:d}'.format(train_step, train_params.NUM_STEPS_TRAIN))
            sys.stdout.flush()  
            
            # Save ckpt periodically
            if train_step % train_params.SAVE_CKPT_STEP == 0:
                self.saver.save(self.sess, self.checkpoint_path, global_step=train_step)
                sys.stdout.write('\nCheckpoint saved.\n')   
                sys.stdout.flush() 
        
        # Stop the agents
        self.stop_agent_event.set()       