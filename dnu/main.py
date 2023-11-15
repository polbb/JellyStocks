# from jellyopt import Portfolio
import jellyopt as js
import optimiser

po = js.Portfolio(['APPL','FB'])
po_2 = js.Portfolio()

print(po.portfolio)
po.add_asset('MMM')
print(po.portfolio)

po = js.Portfolio(['AAA','BBB'])
# print(po.stocks)

po.add_asset('POL')
po.add_asset('LL')
print(po)


optimiser.js(
    cost_fn='sr',
    lower_bound=0,
    upper_bound=1,
    population=20,
    termination=True
)