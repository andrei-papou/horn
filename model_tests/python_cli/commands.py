import os
from time import time
from typing import List, Type

from horn import save_model, save_tensor

from common import Args, SubParsers, get_model_file_path, get_data_file_path, get_model, ModelSpec
from models import model_specs


def _extract_model_spec(args: Args) -> Type[ModelSpec]:
    model_spec = model_specs.get(args.model)
    if model_spec is None:
        raise ValueError(f'Invalid model name: {args.model}')
    return model_spec


class Command:
    name: str

    @classmethod
    def add_cli_parser(cls, sub_parsers: SubParsers):
        parser = sub_parsers.add_parser(name=cls.name)
        parser.set_defaults(func=cls.execute)
        cls._add_cli_arguments(parser)

    @classmethod
    def _add_cli_arguments(cls, parser):
        pass

    @classmethod
    def execute(cls, args: Args):
        raise NotImplementedError()


class SaveModelCommand(Command):
    name: str = 'save_model'

    @classmethod
    def execute(cls, args: Args):
        model_spec = _extract_model_spec(args)
        model = get_model(args.model, model_spec.get_model)
        model_file_path = get_model_file_path(model_name=args.model)
        save_model(model, model_file_path)
        print(f'Model has been saved to "{model_file_path}"')


class SaveDataCommand(Command):
    name: str = 'save_data'

    @classmethod
    def execute(cls, args: Args):
        model_spec = _extract_model_spec(args)
        xs_file_path = get_data_file_path(model_spec.name, 'x')
        ys_file_path = get_data_file_path(model_spec.name, 'y')
        if not os.path.isfile(xs_file_path):
            save_tensor(model_spec.xs, xs_file_path)
            print(f'Saved X data for model "{model_spec.name}" to "{xs_file_path}".')
        else:
            print(f'X data for model `{model_spec.name}` already exists at "{xs_file_path}"')
        if not os.path.isfile(ys_file_path):
            save_tensor(model_spec.ys, ys_file_path)
            print(f'Saved Y data for model "{model_spec.name}" to "{ys_file_path}".')
        else:
            print(f'Y data for model `{model_spec.name}` already exists at "{ys_file_path}"')


class TestCorrectnessCommand(Command):
    name: str = 'test_correctness'

    @classmethod
    def execute(cls, args: Args):
        model_spec = _extract_model_spec(args)
        model = get_model(args.model, model_spec.get_model)
        scores = model.evaluate(x=model_spec.xs, y=model_spec.ys, verbose=0)
        print(f'Keras accuracy for the model `{model_spec.name}` is {scores[1]}')


class TestPerformanceCommand(Command):
    name: str = 'test_performance'

    @classmethod
    def _add_cli_arguments(cls, parser):
        parser.add_argument(
            '-n',
            '--num-iterations',
            dest='num_iterations',
            default=1000,
            type=int,
            help='Number of performance measurement iterations'
        )

    @classmethod
    def execute(cls, args: Args):
        model_spec = _extract_model_spec(args)
        model = get_model(args.model, model_spec.get_model)
        cumulative_time = 0.0
        for _ in range(args.num_iterations):
            start = time()
            model.predict(x=model_spec.xs, verbose=0)
            cumulative_time += time() - start
        print(f'Keras performance for model `{model_spec.name}` : {cumulative_time * 1e6}')


commands: List[Type[Command]] = [
    SaveModelCommand,
    SaveDataCommand,
    TestCorrectnessCommand,
    TestPerformanceCommand,
]
