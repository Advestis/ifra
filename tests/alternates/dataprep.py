from ifra.datapreps import DataPrep


class AlternateDataPrep(DataPrep):
    def dataprep_method(self, x, y):
        x = x.apply(lambda xx: 2 * xx, axis=0)
        return x, y
