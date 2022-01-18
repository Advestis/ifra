from ruleskit import RuleSet

from ifra import Fitter


class AlternateFitter(Fitter):
    def fit(self) -> RuleSet:
        return RuleSet()
