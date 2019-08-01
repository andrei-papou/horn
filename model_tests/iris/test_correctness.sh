echo 'Starting correctness check for model: Iris'

cd python || echo 'No python directory.'
python main.py create_model
python main.py create_testing_data

cd ../rust || echo 'No rust directory.'
cargo run test_correctness
echo 'Completed'
