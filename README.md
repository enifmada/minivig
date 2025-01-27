# minivig
Numpy-based implementation of several popular methods for devigging odds

## Usage
The function `devig_odds` takes as input an n x 2 NumPy array, where each row represents a set of two-way decimal odds to be devigged and the "method" parameter, a string corresponding to the devigging method to be used.
Output is an n x 2 array of devigged decimal odds. Use the auxiliary functions `convert_to_dec_odds` and `convert_to_american` to convert from decimal odds to American odds and vice versa.

`devig_odds` implements the following devigging methods:
- Multiplicative, via `method = "mult"`
- Additive, via `method = "add"`
- Power (using scipy.optimize.minimize to solve for the exponent), via `method = "power"`
- Probit, via `method = "probit"`
- Goto conversion, adapted from [this repository](https://github.com/gotoConversion/goto_conversion), via `method = "goto"`
- Maximization of expected growth, adapted from [this article](https://www.pinnacle.com/en/betting-articles/betting-strategy/why-the-favourite-longshot-bias-is-not-a-bias/qjb2prcq4q96nftd), via `method = "meg"`

An additional utility function, `compute_kelly_frac`, takes as input a market price and a set of two-way devigged decimal odds, and outputs the optimal (full) Kelly Criterion bankroll fraction and the EV of the line.

### Requirements
Numpy, scipy
