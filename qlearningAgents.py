# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        
        """
          Initialize states using the Counter() dictionary datastructure.
          This will indicate the Q Value for a specific state.
          A Q-Value can be found using the (state, action) pair as a key.
        """
        
        # Initializing states as a Python dictionary using util.Counter()
        self.states = util.Counter()

    def getQValue(self, state, action):
                
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        
        if not self.states[(state, action)]:
            return 0.0
        else:
            return self.states[(state, action)]

    def computeValueFromQValues(self, state):

        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        
        # We only access the Q-Value by using the getQValue() function, as instructed.
        
        # Use getLegalActions() function to collect legal actions of current state
        actions = self.getLegalActions(state)

        # Return '0' if no legal action(s) exist (some terminal state is reached)
        if len(actions) == 0:
            return 0.0
        else:
            maxValue = self.getQValue(state, actions[0])
        
        # Loop over legal actions
        for action in actions:
            
            newValue = self.getQValue(state, action)
            
            # If a better value is found, update!
            if newValue > maxValue:
                maxValue = newValue

        # Return best value found, based on legal actions
        return maxValue

    def computeActionFromQValues(self, state):
        
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        
        # We only access the Q-Value by using the getQValue() function, as instructed.
        
        # Use getLegalActions() function to collect legal actions of current state
        actions = self.getLegalActions(state)
        
        # Return 'None' if no legal action(s) exist (some terminal state is reached)
        if len(actions) == 0:
            return None
        else:
            # Define a local array of state-actions pair, s.t. we can use argMax for this state later
            stateActions = util.Counter();

        # Loop over actions and store associated Q-Values
        for action in actions:

            # Use state-action as a key to store Q-Value of legal action
            stateActions[(state,action)] = self.getQValue(state, action)
        
        # Return the best action using argMax, which picks the highest Q-Value
        return stateActions.argMax()[1]

    def getAction(self, state):

        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        
        # Use epsilon value to get random action or return None if no legal actions exist
        if len(self.getLegalActions(state)) == 0:
            return None
        
        # Define actions array
        actions = self.getLegalActions(state)
        
        # Define a local array of state-actions pair, s.t. we can use argMax for this state later
        stateActions = util.Counter();
        
        for action in actions:
            
            # Use state-action as a key to store Q-Value of legal action
            stateActions[(state,action)] = self.getQValue(state, action)
            
        # Based on epsilon probability, decide for random action
        randomAction = util.flipCoin(self.epsilon)
        
        # If the epsilon probability caused for a random action, return it
        if randomAction:
            return random.choice(actions)
        else:
            # If the epsilon probability caused for the best policy, return it
            return stateActions.argMax()[1]

    def update(self, state, action, nextState, reward):
        
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """

        # Set all known data s.t. we can calculate the new Q-Values
        maxValue = self.states[(state,action)]
        nextMaxValue = self.getValue(nextState)
        discount = self.discount
        learningrate = self.alpha
        
        # Compute new Q-Value using the Q-Learning formula
        newQValue = (1 - learningrate) * maxValue + (learningrate * (reward + discount * nextMaxValue))
        
        # Update Q-Value for current state
        self.states[(state,action)] = newQValue
        
        # Make sure we return something!
        return

    def getPolicy(self, state):
        
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        
        # First, get the features from featExtractor of the current state and action
        features = self.featExtractor.getFeatures(state, action)
        
        # Set initial value to zero
        totalFeaturesWeight = 0;
        
        # Loop over all features and add the weighted feature to the total sum
        for feature in features:
            totalFeaturesWeight += self.weights[feature] * features[feature]
        
        # Return the total sum of weighted features
        return totalFeaturesWeight
        
    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        # We only access the Q-Value by using the getQValue() function, as instructed.
        
        # Get features of current state and action
        features = self.featExtractor.getFeatures(state, action)
        
        # Use the formula to calculate the difference
        difference = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action)
        
        # Update all features accordingly
        for feature in features:
            self.weights[feature] = self.weights[feature] + self.alpha * (difference * features[feature])


    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
