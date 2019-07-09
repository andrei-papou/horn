extern crate horn;
extern crate ndarray;

use std::convert::TryInto;

use horn::{Model, NdArrayBackend};
use ndarray::{array, Array2};

fn run_model() -> Result<(), String> {
    let input: Array2<f64> = array![
        [7.9, 3.8, 6.4, 2.0],
        [5.0, 3.5, 1.3, 0.3],
        [6.0, 2.9, 4.5, 1.5],
    ];
    let model = Model::<NdArrayBackend<f64>>::from_file("../horn-py/artifacts/model.horn")?;
    let output: Array2<f64> = model.run(input.try_into()?)?.try_into()?;

    println!("{:?}", output);
    Ok(())
}

fn main() {
    if let Err(e) = run_model() {
        println!("Error: {}", e);
    }
}
