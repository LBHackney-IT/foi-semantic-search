import pytest
import functions
import test_data


def test_extract_ids():
    url = 'https://foi.infreemation.co.uk/redirect/hackney?id=7684'
    result = functions.extract_id(url)
    assert result is not None
    assert type(result) is int


def test_strip_element():
    assert (
        functions.strip_element(test_data.strip_element_test_input)
        == test_data.strip_element_expected_output
    )


def test_strip_response():
    assert (
        functions.strip_response(test_data.strip_response_test_input)
        == test_data.strip_response_expected_output
    )

