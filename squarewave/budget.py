"""Forward-equivalent (fe) budget constants shared by every optimizer in this repo.

One scored candidate is 1 fe, one gradient call is GRAD_FE. GRAD_FE is the measured ratio of H100 time for
score_and_grad vs score on this oracle (0.45 s vs 0.21 s per call across both services), not GrAdaBeam's own
tism_cost of 1.0, which would understate the gradient's price.
"""

GRAD_FE = 2.1


class BudgetExhausted(Exception):
    pass
