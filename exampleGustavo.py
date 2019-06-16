from backtesting import evaluateHist, evaluateIntr
from strategy import Strategy
from order import Order
from event import Event
from sklearn.externals import joblib
import numpy as np


LONG = -1
NEUTRAL = 0
SHORT = 1


class RSI(Strategy):
    """Trade using RSI oscillator based strategy.

    From:
    https://www.investopedia.com/terms/r/rsi.asp
    https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:relative_strength_index_rsi#calculation
    https://www.reddit.com/r/algotrading/comments/9r8984/rsi_calculation_using_python/
    """

    def __init__(self, period=14, overbought_limit=70,
            oversold_limit=30):
        self.overbought_limit = overbought_limit
        self.oversold_limit = oversold_limit
        self.period = period
        self.prices = []
        self.next_position = NEUTRAL

    def rs(self):
        """Calculates RS (Relative Strength).

        RS = Average Gain in period / Average Loss in period
        """
        if not self.prices:
            return 0

        gains = 0
        losses = 0

        # Prices up to self.period
        # From: https://stackoverflow.com/a/646654
        last_n_prices = self.prices[-self.period:]

        # For each price, if previous price was lower, we had a loss
        # Else, we had a gain
        for i, price in enumerate(last_n_prices, start=1):
            if price > self.prices[i - 1]:
                gains += price
            else:
                losses += price

        avg_gain = gains / len(last_n_prices)
        avg_loss = losses / len(last_n_prices)

        return avg_gain / avg_loss if avg_loss else 0

    def rsi(self):
        """Calculates RSI oscillator.

        RSI = 100 - 100 / (1 + rs)
        """
        return 100 - 100 / (1 + self.rs())

    def push(self, event):
        """Executes market orders (buy or sell) using current strategy
        """
        orders = []
        self.prices.append(event.price[3])
        
        rsi = self.rsi()

        if rsi >= self.overbought_limit:
            if self.next_position == SHORT:
                orders.append(Order(event.instrument, -1, 0))
                orders.append(Order(event.instrument, -1, 0))
            
            elif self.next_position == NEUTRAL:
                orders.append(Order(event.instrument, -1, 0))

            self.next_position = LONG

        elif rsi <= self.oversold_limit:
            if self.next_position == LONG:
                orders.append(Order(event.instrument, 1, 0))
                orders.append(Order(event.instrument, 1, 0))
            
            elif self.next_position == NEUTRAL:
                orders.append(Order(event.instrument, 1, 0))

            self.next_position = SHORT

        return orders


class GustavoStrategy(Strategy):
    
    def __init__(self):
        self.clf = joblib.load("Gustavo/nb_clf.pickle")
        self.last_orders = []
        self.last_event_price = None
        self.last_category = 0
        self.buying = True

    def return_as_category(ret):
        if -np.inf < ret <= -0.01:
            return 0
        elif -0.01 < ret <= -0.005:
            return 1
        elif -0.005 < ret <= 0:
            return 2
        elif 0 < ret <= 0.005:
            return 3
        elif 0.005 < ret <= 0.01:
            return 4
        elif 0.01 < ret < np.inf:
            return 5

    def push(self, event):
        self.last_orders.append([event.price[3]])

        orders = []
        if len(self.last_orders) > 3:
            prediction = self.clf.predict(np.array(self.last_orders[-3:]).reshape(1, -1))
            
            ret = 0
            if self.last_event_price:
                ret = (event.price[3] - self.last_event_price) / self.last_event_price
                ret = return_as_category(ret)

            if self.buying:
                if prediction[0] > ret:
                    orders = [Order(event.instrument, 1, 0)]
                else:
                    orders = [Order(event.instrument, -1, 0), Order(event.instrument, -1, 0)]
                    self.buying = False
            else:
                if prediction[0] > ret:
                    orders = [Order(event.instrument, -1, 0)]
                else:
                    orders = [Order(event.instrument, 1, 0), Order(event.instrument, 1, 0)]
                    self.buying = True
        
        return orders


class MM(Strategy):
    """
    The is a Market Maker to trade PETR3 and PBR stocks. Both stocks
    belong to the brazilian company Petrobras, with PETR3 being the
    ticker for the stock inside B3/Bovespa (brazilian stock market) and
    PBR being its equivalent in NYSE (New York Stock Exchange).
    """

    def __init__(self, spread=50):
        self.petr3 = None
        self.usd = None
        self.spread = spread
        self.last_buy_order = None
        self.last_sell_order = None

    def pbr(self):
        """Calculates PBR value.

        Equation: PBR = (PETR3*F / USDBRL) * ti + tf
        This is a linear equation y = ax + b
        """
        ti = 1.02 # source: linear regression
        tf = -0.3 # source: linear regression
        f = 2 # source: Raul
        return (self.petr3*f / self.usd) * ti + tf

    def push(self, event):
        """Executes market orders (buy or sell) using current strategy
        """

        # Store PETR3 event
        if event.instrument == "PETR3":
            self.petr3 = event.price[3]
        
        # Store USDBRL event
        if event.instrument == "USDBRL":
            self.usd = event.price[3]
        
        # Once we have both, we can calculate PBR value
        if self.petr3 and self.usd:
            pbr = self.pbr()
            
            # If a buy order already exists, we need to cancel it
            # in order to create our new position
            if self.last_buy_order:
                self.cancel(self.id, self.last_buy_order)

            # Same for a sell order
            if self.last_sell_order:
                self.cancel(self.id, self.last_sell_order)
            
            buy = Order("PBR", 1, pbr - self.spread)
            self.last_buy_order = buy.id
            
            sell = Order("PBR", -1, pbr + self.spread)
            self.last_sell_order = sell.id
            
            return [buy, sell]

        # If we don't have PETR3 or USDBRL values, do nothing
        return []

    def fill(self, instrument, price, quantity, status):
        super().fill(instrument, price, quantity, status)

        # If PBR is being bought, I need to sell PETR3 (to acquire BRL)
        # and need to buy USD (with the BRL acquired). With the USD
        # I can buy PBR to offer it.
        if quantity > 0:
            orders = [Order("PETR3", -1, 0), Order("USDBRL", 1, 0)]

        # If PBR is being offered, I need to buy BRL (sell USD). 
        else:
            orders = [Order("PETR3", 1, 0), Order("USDBRL", -1, 0)]

        return orders


print(evaluateHist(RSI(), {'IBOV': '^BVSP.csv'}))
print(evaluateHist(GustavoStrategy(), {'IBOV': '^BVSP.csv'}))
print(evaluateIntr(MM(), {
    'USDBRL':'./USDBRL.csv',
    'PETR3':'./PETR3.csv',
    'PBR': None
}))
