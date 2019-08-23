from typing import List

from horn.exceptions import NotSupportedException


SUPPORTED_PADDING_OPTS = [u'same', u'valid']
SUPPORTED_DATA_FORMAT_OPTS = [u'channels_first', u'channels_last']


def _validate_str_choices(name, value, supported_values):
    # type: (str, str, List[str]) -> None
    if value not in supported_values:
        raise NotSupportedException(
            u'{} "{}" is not supported. The supported ones: {}'.format(name, value, supported_values)
        )


def validate_padding(padding):
    # type: (str) -> None
    _validate_str_choices('Padding', padding, SUPPORTED_PADDING_OPTS)


def validate_data_format(padding):
    # type: (str) -> None
    _validate_str_choices('Data format', padding, SUPPORTED_DATA_FORMAT_OPTS)
