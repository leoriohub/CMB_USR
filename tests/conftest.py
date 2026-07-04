"""Test suite configuration for CMB Anomaly project.

Makes project root importable and provides shared fixtures.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np


@pytest.fixture
def higgs_model():
    from models.higgs import HiggsModel
    model = HiggsModel()
    model.x0 = 5.70
    model.y0 = -0.10
    return model


@pytest.fixture
def ezquiaga_model():
    from models.ezquiaga_chi import EzquiagaCHIModel
    model = EzquiagaCHIModel()
    return model


@pytest.fixture
def planck_data():
    from scripts.planck_data import get_planck_data_asymmetric
    return get_planck_data_asymmetric()
