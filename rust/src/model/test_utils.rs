use std::convert::TryFrom;
use std::process;

use crate::backends::Backend;
use crate::common::types::{HError, HResult};

enum TestCommand {
    Correctness,
    Performance,
}

pub trait TestCommandRunner {
    type Backend: Backend;

    fn test_correctness() -> HResult<()>;
    fn test_performance() -> HResult<()>;
}

impl TestCommand {
    const CORRECTNESS: &'static str = "test_correctness";
    const PERFORMANCE: &'static str = "test_performance";

    fn execute<R>(&self) -> HResult<()>
    where
        R: TestCommandRunner,
        R::Backend: Backend,
    {
        match self {
            TestCommand::Correctness => R::test_correctness(),
            TestCommand::Performance => R::test_performance(),
        }
    }
}

impl TryFrom<&str> for TestCommand {
    type Error = HError;

    fn try_from(value: &str) -> HResult<TestCommand> {
        match value {
            TestCommand::CORRECTNESS => Ok(TestCommand::Correctness),
            TestCommand::PERFORMANCE => Ok(TestCommand::Performance),
            _ => Err(format_err!("Unknown command: {}", value)),
        }
    }
}

fn handle_error(error: &str) {
    eprintln!("{}", error);
    process::exit(1);
}

fn handle_test_command_impl<R>(argv: &Vec<String>) -> Result<(), String>
where
    R: TestCommandRunner,
    R::Backend: Backend,
{
    let cmd_name = match argv.iter().skip(1).next() {
        Some(v) => v,
        None => return Err(String::from("Command name is not provided.")),
    };
    let cmd =
        TestCommand::try_from(cmd_name.as_str()).map_err(|_| String::from("Invalid command."))?;
    cmd.execute::<R>().map_err(|err| format!("{}", err))
}

pub fn handle_test_command<R>(argv: &Vec<String>)
where
    R: TestCommandRunner,
    R::Backend: Backend,
{
    if let Err(error) = handle_test_command_impl::<R>(argv) {
        handle_error(error.as_str());
    }
}
