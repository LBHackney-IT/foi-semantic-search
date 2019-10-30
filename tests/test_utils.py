from foi_semantic_search import utils
import test_data
import os
import sys

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../foi_semantic_search/')


def test_extract_ids():
    url = 'https://foi.infreemation.co.uk/redirect/hackney?id=7684'
    result = utils.extract_id(url)
    assert result is not None
    assert type(result) is int


def test_strip_element():
    assert (
        utils.strip_element(test_data.strip_element_test_input)
        == test_data.strip_element_expected_output
    )


def test_strip_response():
    assert (
        utils.strip_response(test_data.strip_response_test_input)
        == test_data.strip_response_expected_output
    )

