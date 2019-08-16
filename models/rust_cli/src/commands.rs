use clap::{App, Arg, ArgMatches, SubCommand};
use horn::{Backend, FromFile, HError, HResult, Model, OneHotMax};
use std::convert::TryInto;
use std::time::Instant;

use crate::models::TestModel;

pub trait Command {
    const NAME: &'static str;

    fn execute<M: TestModel>(matches: &ArgMatches) -> HResult<()>
    where
        <M::Backend as Backend>::CommonRepr: TryInto<M::Y, Error = HError>;

    fn command_args_getter<'a, 'b>() -> App<'a, 'b>;
}

pub fn command_handle<C, M>(model_name: &str, matches: &ArgMatches) -> HResult<()>
where
    C: Command,
    M: TestModel,
    <M::Backend as Backend>::CommonRepr: TryInto<M::Y, Error = HError>,
{
    if let Some(matches) = matches.subcommand_matches(C::NAME) {
        if model_name == M::NAME {
            return C::execute::<M>(matches);
        }
    }
    Ok(())
}

pub fn run_cli<'a, 'b>(
    cmd_args: Vec<App<'a, 'b>>,
    cmd_variants: Vec<fn(&str, &ArgMatches) -> HResult<()>>,
) -> HResult<()> {
    let app = App::new("Horn Rust CLI").arg(
        Arg::with_name("model")
            .short("m")
            .long("model")
            .help("Model to run the command for")
            .required(true)
            .takes_value(true),
    );
    let app = cmd_args
        .into_iter()
        .fold(app, |app, subcmd| app.subcommand(subcmd));
    let matches = app.get_matches();

    let model = match matches.value_of("model") {
        Some(m) => m,
        None => return Err(format_err!("Model name is required (--model)")),
    };

    for cmd_var in cmd_variants {
        cmd_var(model, &matches)?;
    }

    Ok(())
}

pub struct TestCorrectnessCommand;

impl Command for TestCorrectnessCommand {
    const NAME: &'static str = "test_correctness";

    fn execute<M: TestModel>(_matches: &ArgMatches) -> HResult<()>
    where
        <M::Backend as Backend>::CommonRepr: TryInto<M::Y, Error = HError>,
    {
        let model_file_path = M::get_model_file_path();
        let xs_file_path = M::get_xs_file_path();
        let ys_file_path = M::get_ys_file_path();

        let model = Model::<M::Backend>::from_file(model_file_path.as_str())?;
        let xs = M::X::from_file(xs_file_path.as_str())?;
        let ys = M::Y::from_file(ys_file_path.as_str())?;
        let xs_cr = <M::X as TryInto<<M::Backend as Backend>::CommonRepr>>::try_into(xs)?;

        let ys_hat_cr = model.run(xs_cr)?;
        let ys_hat: M::Y =
            <<M::Backend as Backend>::CommonRepr as TryInto<M::Y>>::try_into(ys_hat_cr)?;

        let accuracy = M::get_accuracy(&ys, &ys_hat.one_hot_max(1)?)?;
        println!(
            "Horn accuracy for model `{}` is {:?}. Backend: `{}`",
            M::NAME,
            accuracy,
            M::Backend::NAME
        );

        Ok(())
    }

    fn command_args_getter<'a, 'b>() -> App<'a, 'b> {
        SubCommand::with_name(Self::NAME)
            .about("Run correctness check by calculating accuracy.")
            .version("0.0.1")
    }
}

pub struct TestPerformanceCommand;

impl TestPerformanceCommand {
    const DEFAULT_NUM_ITERATIONS: usize = 1000;
}

impl Command for TestPerformanceCommand {
    const NAME: &'static str = "test_performance";

    fn execute<M: TestModel>(matches: &ArgMatches) -> HResult<()>
    where
        <M::Backend as Backend>::CommonRepr: TryInto<M::Y, Error = HError>,
    {
        let model_file_path = M::get_model_file_path();
        let xs_file_path = M::get_xs_file_path();

        let model = Model::<M::Backend>::from_file(model_file_path.as_str())?;
        let xs = M::X::from_file(xs_file_path.as_str())?;

        let num_iterations: usize = matches
            .value_of("num-iterations")
            .map(|x| x.parse::<usize>())
            .unwrap_or(Ok(Self::DEFAULT_NUM_ITERATIONS))?;

        let mut cumulative_time: u128 = 0;
        for _ in 0..num_iterations {
            let data =
                <M::X as TryInto<<M::Backend as Backend>::CommonRepr>>::try_into(xs.clone())?;
            let timer = Instant::now();
            model.run(data)?;
            cumulative_time += timer.elapsed().as_micros();
        }

        println!(
            "Horn performance for model `{}` is {:?}. Backend: `{}`",
            M::NAME,
            cumulative_time,
            M::Backend::NAME
        );

        Ok(())
    }

    fn command_args_getter<'a, 'b>() -> App<'a, 'b> {
        SubCommand::with_name(Self::NAME)
            .about("Test performance by running the model a number of times.")
            .version("0.0.1")
            .arg(
                Arg::with_name("num-iterations")
                    .short("n")
                    .long("num-iterations")
                    .takes_value(true)
                    .help("Number of accuracy calculation iterations."),
            )
    }
}
