from ruleskit import RuleSet

from ifra import Fitter


class AlternateFitter(Fitter):
    def fit(self, x, y) -> RuleSet:
        return RuleSet()
