echo 'Starting correctness check for model: MNIST MLP'

cd python || echo 'No python directory.'
# Prepare model and data files
python main.py create_model
python main.py create_test_data

# Run python correctness check
python main.py test_correctness

cd ../rust || echo 'No rust directory.'
# Run rust correctness check
cargo --quiet run --release test_correctness