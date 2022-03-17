import pandas as pd
from transparentpath import TransparentPath


def load_y(path: TransparentPath, **kwargs) -> pd.Series:
    y = path.read(**kwargs).squeeze()
    if isinstance(y, pd.Series):
        return y
    else:
        return pd.Series([y])
