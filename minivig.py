import numpy as np
from scipy.optimize import minimize as spoptmin
from scipy.stats import norm as sps_norm

def convert_to_dec_odds(am_odds):
    if np.any(np.abs(am_odds) > 99):
        return np.where(am_odds < 0, 1 - 100 / am_odds, 1 + am_odds / 100)
    else:
        return sb_odds

def convert_to_american(dec_odds):
    return np.where(dec_odds > 2, (dec_odds - 1) * 100, -100 / (dec_odds - 1))


def power_devig_fn(k, odds):
    return np.abs(np.sum(1 / np.power(odds, 1 / k)) - 1)


def devig_odds(odds, method="add"):
    # devigs an n x 2 array of DECIMAL odds. returns devigged decimal odds.
    # use convert_to_dec_odds first if your odds are American.
    assert odds.shape[1] == 2
    if method == "mult":
        odds_sum = np.sum(odds, axis=1)
        return 1 / (odds[:, ::-1] / odds_sum[:, np.newaxis])
    elif method == "add":
        inv_odds_sum = np.sum(1 / odds, axis=1)
        factor = (inv_odds_sum - 1) / 2
        return 1 / np.clip(1 / odds - factor[:, np.newaxis], 1e-15, 1 - 1e-15)
    elif method == "goto":
        probs = 1 / odds
        errors = np.sqrt(probs * (1 - probs) / probs)
        step_size = (np.sum(probs) - 1) / np.sum(errors)
        return 1 / np.clip(probs - errors * step_size, 1e-15, 1 - 1e-15)
    elif method == "power":
        best_ratio = spoptmin(power_devig_fn, x0=1, args=(odds))
        pow_odds = 1 / np.power(odds, 1 / best_ratio["x"])
        if np.isclose(np.sum(pow_odds), 1):
            return 1 / pow_odds
        else:
            raise ValueError("failure in power method!")
    elif method == "meg":
        probs = 1 / odds
        meg_odds = np.zeros_like(odds)
        meg_odds[:, 0] = np.log(probs[:, 0] / (1 - probs[:, 1])) / np.log(probs[:, 1] / (1 - probs[:, 0]))
        meg_odds[:, 1] = 1 / meg_odds[:, 0]
        meg_odds += 1
        return meg_odds
    elif method == "probit":
        probs = 1 / odds
        probits = sps_norm.ppf(probs)
        probits -= np.sum(probits, axis=1) / 2
        return 1 / sps_norm.cdf(probits)
    else:
        raise NotImplementedError("invalid odds fixing method!")


def compute_kelly_frac(price, devigged_odds):
    #compute the optimal fully kelly bankroll fraction as a function of
    #an offered price and a set of devigged odds.
    #returns the full kelly fraction and the play's EV.
    price_odds = convert_to_dec_odds(np.array([price]))
    probs = 1 / devigged_odds
    probs /= np.sum(probs)
    e_money = price_odds.tolist()[0] * probs[0]
    ev = 100 * (e_money - 1)
    return probs[0] - probs[1] / (price_odds[0] - 1), ev