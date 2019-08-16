#[macro_use]
extern crate failure;

mod commands;
mod models;

use horn::NdArrayBackend;

use commands::{command_handle, run_cli, Command, TestCorrectnessCommand, TestPerformanceCommand};
use models::{IrisModel, MnistMLPModel};

fn main() {
    if let Err(error) = run_cli(
        vec![
            TestPerformanceCommand::command_args_getter(),
            TestCorrectnessCommand::command_args_getter(),
        ],
        vec![
            command_handle::<TestCorrectnessCommand, IrisModel<NdArrayBackend<f64>>>,
            command_handle::<TestCorrectnessCommand, MnistMLPModel<NdArrayBackend<f64>>>,
            command_handle::<TestPerformanceCommand, IrisModel<NdArrayBackend<f64>>>,
            command_handle::<TestPerformanceCommand, MnistMLPModel<NdArrayBackend<f64>>>,
        ],
    ) {
        println!("Error: {}", error.as_fail());
    };
}
