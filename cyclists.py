import numpy as np

"""
This is a script that contains the logic to model an agent-based simulation
of an Evolutionary Game for the Brakaway Dilemma, where a cyclist chooses between two strategies:
to break away from the rest of the peloton (attack) or to stay and cooperate with the rest (no attack)

Each cyclist has to take into account two payoff matrices, realted to each strategy combination 
between two players: a first payoff matrix that models the distance it gains from the peloton, and
a second payoff matrix that models the energy expenditure of choosing that strategy.

In this model we have two dynamics: the first is the cyclist's physical dynamics i.e. how much distance it travels
given that it chooses a particular strategy and how that affects the energy expenditure on choosing one 
strategy or the other ( work against air resistance, which is proportional to it's speed and how much extra work it 
has to do if it chooses to attack). If cyclist chooses to not attack, it will only exert the needed
force to keep going at a certain speed, which is the same as the peloton's speed at the beginning.
If it chooses to attack, it wil exert an extra constant force that will make it gain distance
from the peloton.

The second dynamic is related with the cyclist's decision making process. The cyclist will choose to
attack or not taking into account the average payoff of each cyclist in the peloton, compared to
it's own current payoff. The cyclist will choose to attack with a higher probability if the average
payoff of the rest of the group if higher than it's own payoff, if not it will choose not to attack.
Now, this probability is a conditional probability, given the current remaining stamina of the cyclist.
In that idea, each cyclist will have an initial stamina that will constantly be drained out due to
air resistance. If the cyclist chooses to attack, it will drain an extra amount of stamina, on top
of what is drained out by air resistance. If the cyclist's current stamina reaches values below
cero, it will not be able to attack anymore (probability near zero) and will only choose not attack.
"""

class Cyclist:
    def __init__(self, stamina, x0, strat, dt, gamma, des_threshold = 50, attack_v = 12.5, no_attack_v = 7):
        self.stamina = stamina
        self.init_stamina = stamina
        self.x0 = x0
        self.x1 = x0
        self.strat = strat #1 for attack, 0 for no attack
        self.gamma = gamma
        self.attack_v = attack_v
        self.no_attack_v = no_attack_v
        self.des_threshold = des_threshold
        self.dt = dt
        self.w = 0 #initial work
        #choosing initial speed and force
        if strat == 1:
            self.x2 = self.attack_v #m/s
            self.f = self.attack_v*self.gamma #N/kg
        else:
            self.x2 = self.no_attack_v #m/s
            self.f = self.no_attack_v*self.gamma #N/kg
    def step(self):
        if self.strat == 1:
            if self.stamina <= 0:
                self.f = 0.2*self.no_attack_v*self.gamma #N/kg
            else:
                self.f = self.attack_v*self.gamma #N/kg
            self.x2 += (self.f-self.gamma*self.x2)*self.dt
        else:
            if self.stamina <= 0:
                self.f = 0.2*self.no_attack_v*self.gamma #N/kg
            else:
                self.f = self.no_attack_v*self.gamma #N/kg
            self.x2 += (self.f-self.gamma*self.x2)*self.dt
            self.stamina += 0.5*self.f*(self.x2*self.dt)
        self.x1 += self.x2*self.dt
        self.w += self.f*(self.x2*self.dt)
        self.stamina -= self.f*(self.x2*self.dt)
    def update_strat(self, payoff, avg_payoff,
                 noise=1,
                 k=30,
                 stamina_threshold=0.2):
        # ========================================
        # Remaining stamina fraction
        # ========================================

        s = (self.stamina) / self.init_stamina

        # ========================================
        # Saturated sigmoid helper
        # ========================================

        def sigmoid(x):
            # Prevent overflow in exp()
            x = np.clip(x, -60, 60)
            return 1 / (1 + np.exp(-x))

        # ========================================
        # Smooth stamina gate
        # ========================================

        gate = sigmoid(k * (s - stamina_threshold))

        # ========================================
        # Baseline payoff-driven probability
        # ========================================

        payoff_arg = (avg_payoff - payoff - self.des_threshold) / noise
        p_base = sigmoid(payoff_arg)

        # ========================================
        # Transition probabilities
        # ========================================

        # NA -> A suppressed by exhaustion
        p_A = gate * p_base

        # A -> NA enhanced by exhaustion
        p_N = (1 - gate) + gate * (1 - p_base)

        # ========================================
        # Strategy update
        # ========================================

        if self.strat == 0:
            if np.random.rand() < p_A:
                self.strat = 1

        else:
            if np.random.rand() < p_N:
                self.strat = 0
        
        