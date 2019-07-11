extern crate horn;
extern crate ndarray;

use std::convert::TryInto;

use horn::{Model, NdArrayBackend, HResult};
use ndarray::{array, Array2};

fn run_model() -> HResult<()> {
    let input: Array2<f64> = array![
        [5.1, 3.5, 1.4, 0.2],
        [5.5, 2.4, 3.7, 1. ],
        [5.3, 3.7, 1.5, 0.2],
    ];
    let model: Model<NdArrayBackend<f64>> = Model::from_file("../horn-py/artifacts/model.horn")?;
    let output: Array2<f64> = model.run(input.try_into()?)?.try_into()?;

    println!("{:?}", output);
    Ok(())
}

fn main() {
    if let Err(e) = run_model() {
        println!("Error: {}", e.to_string());
    }
}
