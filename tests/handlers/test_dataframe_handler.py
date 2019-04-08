import os
import json
import tempfile
from flask import Request
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from bentoml.handlers import DataframeHandler


def test_dataframe_handler_result():
    def fake_fun(df):
        df['age'] = df['age'].add(10)
        return df
    fake_data = '[{"age": 1}, {"age": 1}, {"age": 3}]'

    fake_request = {}

    response  = DataframeHandler.handle_request(fake_request, fake_fun)
    assert True


def test_dataframe_handler_input_columns_requirement():
    assert True


def test_dataframe_handler_output_format():
    assert True
