use std::convert::TryInto;
use std::env;
use std::time::Instant;

use horn::{Model, FromFile, NdArrayBackend, HResult, model_test_utils, OneHotMax, model_evaluation};
use ndarray::{Array2};

struct NdArrayRunner;

impl model_test_utils::TestCommandRunner for NdArrayRunner {
    type Backend = NdArrayBackend<f64>;

    fn test_correctness() -> HResult<()> {
        let xs: Array2<f64> = Array2::from_file("../artifacts/x.data")?;
        let ys: Array2<f64> = Array2::from_file("../artifacts/y.data")?;
        let model: Model<NdArrayBackend<f64>> = Model::from_file("../artifacts/mnist_mlp.model")?;
        let output: Array2<f64> = model.run(xs.try_into()?)?.try_into()?;
        let output = output.one_hot_max(1)?;

        println!("Horn accuracy: {:?}", model_evaluation::dim2::accuracy(&ys, &output)?);
        Ok(())
    }

    fn test_performance() -> HResult<()> {
        let xs: Array2<f64> = Array2::from_file("../artifacts/x.data")?;
        let model: Model<NdArrayBackend<f64>> = Model::from_file("../artifacts/mnist_mlp.model")?;

        let mut cumulative_time: u128 = 0;
        for _ in 0..1000 {
            let data = xs.clone();
            let timer = Instant::now();
            model.run(data.try_into()?)?;
            cumulative_time += timer.elapsed().as_micros();
        }
        println!("Horn performance: {}", cumulative_time);

        Ok(())
    }
}

fn main() {
    let argv: Vec<String> = env::args().into_iter().collect();

    model_test_utils::handle_test_command::<NdArrayRunner>(&argv);
}
