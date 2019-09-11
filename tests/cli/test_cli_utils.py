import pytest
import click

from bentoml.cli.click_utils import parse_bento_tag_callback


def test_parse_bento_tag_callback():
    test_good_pattern = 'mybento:v1.23.0'
    result = parse_bento_tag_callback(None, None, test_good_pattern)
    assert result == 'mybento:v1.23.0'

    test_bad_pattern = 'bad_pattern'
    with pytest.raises(click.BadParameter) as e:
        parse_bento_tag_callback(None, None, test_bad_pattern)

    assert str(e.value).startswith('Bad formatting.')
