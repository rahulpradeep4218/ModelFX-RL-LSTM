import Data_Process as FXDataProcess
import ForexUtils as FXUtils
import gym
from gym import  spaces
import numpy as np


# Position Constant :
LONG = 0
SHORT = 1
FLAT = 2

# Action Constant :
BUY = 0
SELL = 1
HOLD = 2


class ForexEnv(gym.Env):
    def __init__(self, inputSymbol, show_trade=True, type="train"):
        self.showTrade = show_trade
        self.actions = ["LONG", "SHORT", "FLAT"]
        self.symbol = inputSymbol
        self.type = type
        self.sqlEngine = FXUtils.getSQLEngine()
        ml_variables = FXUtils.getMLVariables()
        self.variables = ml_variables
        self.spread = float(ml_variables['Spread'])
        self.window_size = int(ml_variables['WindowSize'])

        ### Balance variables
        self.balance = float(self.variables['StartingBalance'])
        self.starting_balance = self.balance
        self.portfolio = float(self.balance)
        self.minPortfolio = self.portfolio


        data = FXDataProcess.ProcessedData(inputSymbol, train_test_predict=type)
        data.addSimpleFeatures()
        data.apply_normalization()
        self.df = data.df.values
        self.rawData = data.rawData
        self.split_point = data.split_point
        # Features
        self.n_features = self.df.shape[1]
        self.shape = (self.window_size, self.n_features)

        # Action space and Observation space
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(high=np.inf, low=-np.inf, shape=self.shape, dtype=np.float64)

        # Profit Calculation Variables
        self.OnePip = float(ml_variables['SymbolOnePip'])
        self.PipVal = float(ml_variables['SymbolPipValue'])
        self.MaxLossPerTrade = float(ml_variables['MaxLossPerTrade'])
        self.BoxPips = float(ml_variables['PipsDivision'])
        self.RiskRewardRatio = float(ml_variables['RiskRewardRatio'])



    def render(self, mode='human', verbose=False):
        return None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):

        if self.done:
            return self.state, self.reward, self.done, {}

        #print("Action : ",action)
        self.action = HOLD
        self.reward = 0.0
        self.current_price = self.getPrice(self.current_tick)

        if action == BUY:
            if self.position == FLAT or self.position == LONG:
                self.position = LONG
                self.action = BUY
                self.entryPrices.append(self.current_price)
                self.stopLosses.append(self.current_price - (self.BoxPips * self.OnePip))
                #self.takeProfit = self.entryPrice + (self.BoxPips * self.OnePip * self.RiskRewardRatio)
                self.portfolio = self.balance + self.calculateTotalProfit()
                if self.type == "test":
                    FXUtils.execute_query_db("INSERT INTO metaactions(Action,Time) VALUES('BUY','" + str(self.rawData.index[self.current_tick]) + "')", self.sqlEngine)

            elif self.position == SHORT and self.variables['UseTakeProfit'] != 'true':
                self.action = BUY
                profit = self.calculateTotalProfit()
                # self.reward += profit
                self.balance = self.balance + profit
                self.reward += profit
                for i, entryPrice in enumerate(self.entryPrices):
                    self.sells += 1
                    profit = self.calculateProfit(index=i)
                    if profit < 0:
                        self.lostSells += 1
                self.entryPrices = []
                self.stopLosses = []
                self.position = FLAT

        elif action == SELL:
            if self.position == FLAT:
                self.action = SELL
                self.position = SHORT
                self.entryPrices.append(self.current_price)
                self.stopLosses.append(self.current_price + (self.BoxPips * self.OnePip))
                #self.takeProfit = self.entryPrice - (self.BoxPips * self.OnePip * self.RiskRewardRatio)
                if self.type == "test":
                    FXUtils.execute_query_db("INSERT INTO metaactions(Action,Time) VALUES('SELL','" + str(self.rawData.index[self.current_tick]) + "')", self.sqlEngine)

            elif self.position == LONG and self.variables['UseTakeProfit'] != 'true':
                self.action = SELL
                profit = self.calculateTotalProfit()
                self.reward += profit
                # self.reward += profit
                self.balance = self.balance + profit
                for i, entryPrice in enumerate(self.entryPrices):
                    self.buys += 1
                    profit = self.calculateProfit(index=i)
                    if profit < 0:
                        self.lostBuys += 1
                self.position = FLAT
                self.entryPrices = []
                self.stopLosses = []


        self.updateBalance()

        #self.reward = self.calculate_reward()
        #self.reward = (self.reward + self.balance - self.starting_balance) / self.starting_balance
        self.reward = (self.reward) / self.starting_balance
        self.current_tick += 1

        if self.showTrade and self.current_tick % 1000 == 0:
            print("Tick : {0} / Portfolio : {1} / Balance : {2} / Min Portfolio : {3}".format(self.current_tick, self.portfolio, self.balance, self.minPortfolio))
            print("buys : {0} / {1} , sells : {2} / {3}".format((self.buys - self.lostBuys), self.lostBuys, (self.sells - self.lostSells), self.lostSells))
        self.history.append((self.action, self.current_tick, self.current_price, self.portfolio, self.reward))

        self.updateState()

        if self.current_tick > (self.df.shape[0] - self.window_size - 1):
            self.done = True
            # self.reward = self.calculate_reward()

        return self.state, self.reward, self.done, {'portfolio': np.array([self.portfolio]),
                                                    'buys': self.buys, 'lostBuys': self.lostBuys,
                                                    'sells': self.sells, 'lostSells': self.lostSells}



    def updateState(self):
        one_hot_position = FXUtils.getOneHotEncoding(self.position, 3)
        #profit = self.calculateProfit()
        #profit = self.portfolio - self.starting_balance
        #print("df : ", self.df[self.current_tick], " one hot : ",one_hot_position)

        # self.state = np.concatenate((self.df[self.current_tick], one_hot_position))
        self.state = self.df[self.current_tick]
        return self.state

    def updateStopLoss(self, index):

        curr_price = self.stopLosses[index]
        entryPrice = self.entryPrices[index]
        if self.position == LONG:
            profit = (((curr_price - entryPrice) / self.OnePip) - self.spread) * self.calculateQuanity() * self.PipVal
            self.buys += 1
            self.lostBuys += 1
            self.balance += profit

        elif self.position == SHORT:
            profit = (((entryPrice - curr_price) / self.OnePip) - self.spread) * self.calculateQuanity() * self.PipVal
            self.sells += 1
            self.lostSells += 1
            self.balance += profit

        self.reward += profit


    def updateBalance(self):
        toDelete = []
        if self.position == LONG:

            for i, stopLoss in enumerate(self.stopLosses):
                if self.getPrice(self.current_tick, type='L') <= stopLoss:
                    self.updateStopLoss(i)
                    toDelete.append(i)


        elif self.position == SHORT:
            for i, stopLoss in enumerate(self.stopLosses):
                if self.getPrice(self.current_tick, type='H') >= stopLoss:
                    self.updateStopLoss(i)
                    toDelete.append(i)

        self.stopLosses = FXUtils.deleteFromList(x=self.stopLosses, indexList=toDelete)
        self.entryPrices = FXUtils.deleteFromList(x=self.entryPrices, indexList=toDelete)

        if len(self.stopLosses) == 0:
            self.position = FLAT

        #### Update of Portfolio
        self.portfolio = self.balance + self.calculateTotalProfit()

        #### Check minimum portfolio
        if self.portfolio < self.minPortfolio:
            self.minPortfolio = self.portfolio


    def calculate_reward(self):

        if self.position == FLAT:
            reward = 0
        else:
            reward = self.calculateTotalProfit() / self.starting_balance
        if self.portfolio > self.max_portfolio:
            self.max_portfolio = self.portfolio
        return reward - stepReward

    def getPrice(self, index,type = 'C'):
        if type == 'O':
            return self.rawData.iloc[index]['Open']
        elif type == 'H':
            return self.rawData.iloc[index]['High']
        elif type == 'L':
            return self.rawData.iloc[index]['Low']
        elif type == 'C':
            return self.rawData.iloc[index]['Close']

    def calculateTotalProfit(self, type='C'):

        if type == 'O' or type == 'H' or type =='L' or type == 'C':
            curr_price = self.getPrice(self.current_tick, type=type)
        elif type == 'T':
            curr_price = self.takeProfit
        elif type == 'S':
            curr_price = self.stopLoss

        profit = 0.0
        if self.position == LONG:
            for entryPrice in self.entryPrices:
                profit += (((curr_price - entryPrice) / self.OnePip) - self.spread) * self.calculateQuanity() * self.PipVal
        elif self.position == SHORT:
            for entryPrice in self.entryPrices:
                profit += (((entryPrice - curr_price) / self.OnePip) - self.spread) * self.calculateQuanity() * self.PipVal

        return profit


    def calculateProfit(self, index, type='C'):
        if type == 'O' or type == 'H' or type =='L' or type == 'C':
            curr_price = self.getPrice(self.current_tick, type=type)
        entryPrice = self.entryPrices[index]
        if self.position == LONG:
            profit = (((curr_price - entryPrice) / self.OnePip) - self.spread) * self.calculateQuanity() * self.PipVal
        elif self.position == SHORT:
            profit = (((entryPrice - curr_price) / self.OnePip) - self.spread) * self.calculateQuanity() * self.PipVal

        return profit

    def calculateQuanity(self):
        return self.MaxLossPerTrade / (self.PipVal * float(self.variables['PipsDivision']))

    def reset(self):
        self.current_tick = 0
        print("Starting Episode : ")

        # Positions
        self.buys = 0
        self.sells = 0
        self.lostBuys = 0
        self.lostSells = 0


        # Clear the variables
        self.balance = float(self.variables['StartingBalance'])
        self.starting_balance = self.balance
        self.portfolio = float(self.balance)
        self.minPortfolio = self.portfolio
        self.profit = 0
        self.max_portfolio = self.starting_balance

        self.action = HOLD
        self.position = FLAT
        self.done = False

        self.history = []
        self.entryPrices = []
        self.stopLosses = []

        self.updateState()
        #print("State : ",self.state)
        return self.state
