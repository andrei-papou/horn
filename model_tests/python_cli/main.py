from argparse import ArgumentParser

from commands import commands


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        '-m', '--model', dest='model', type=str, required=True, help='A model to run the command for'
    )
    subparsers = arg_parser.add_subparsers()
    for cmd in commands:
        cmd.add_cli_parser(subparsers)

    args = arg_parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
