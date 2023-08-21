import os
import ivy
import pytest
import numpy as np
from ivy_models_tests import helpers
from ivy_models.inceptionnet import func2

func2()


def test_func():
    assert func2() == 5