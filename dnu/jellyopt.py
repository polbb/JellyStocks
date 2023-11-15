__all__ = ['Portfolio']  # This sets the available methods and functions that will be imported when importing all

def dummy():
    print('testing')


class Portfolio:
    risk_free = 0

    def __init__(self, assets_list=None, amount=None):
        self.portfolio = assets_list
        self.amount = amount

        # self.number_of_assets = len(self.portfolio)

    def add_asset(self, asset):
        self.portfolio.append(asset)

    @staticmethod
    def print():
        dummy()

    @classmethod
    def set_risk_free_rate(cls, rate):
        cls.risk_free = rate

    def __str__(self):
        return f'{self.portfolio}'


# po = Portfolio(['APPL', 'FB'])
# po_2 = Portfolio()
#
# print(po.portfolio)
# po.add_asset('MMM')
# print(po.portfolio)
#
# po = Portfolio(['AAA', 'BBB'])
# # print(po.stocks)
#
# po.add_asset('POL')
# po.add_asset('LL')
# print(po)
