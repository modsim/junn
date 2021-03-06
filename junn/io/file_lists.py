"""File list helper functions."""
import re


def generate_glob_and_replacer(search, replace):
    """
    Prepare a wildcard pattern for globbing and replacing.

    :param search:
    :param replace:
    :return:
    """
    replace = generate_replacer(search, replace)
    glob_pattern = prepare_for_regex(search, task='glob')
    return glob_pattern, replace


def generate_replacer(search, replace):
    """
    Prepare a wildcard pattern to be used a replacement regex.

    :param search:
    :param replace:
    :return:
    """
    search = prepare_for_regex(search)
    replace = prepare_for_regex(replace, task='replace')

    search = re.compile(search)

    def _inner(pattern):
        return search.sub(replace, pattern)

    return _inner


def _prepare_for_regex_get_processor(task):
    if task == 'search':

        def process(_, split):
            return re.escape(split) + '(.*)'

    elif task == 'replace':

        def process(n, split):
            return split + '\\%d' % (n,)

    elif task == 'glob':

        def process(_, split):
            return split + '*'

    else:
        raise RuntimeError("Unsupported task.")

    return process


def prepare_for_regex(input_, task='search'):
    """
    Prepare a wildcard pattern for specific use.

    :param input_:
    :param task:
    :return:
    """
    splits = input_.split('{}')

    process = _prepare_for_regex_get_processor(task)

    return (
        ''.join(
            process(n, split)
            for n, split in zip([n for n, _ in enumerate(splits)][1:], splits)
        )
        + splits[-1]
    )
