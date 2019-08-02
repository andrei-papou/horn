echo 'Starting performance check for model: MNIST MLP'

cd python || echo 'No python directory.'
python main.py create_model
python main.py create_test_data
python main.py test_performance

cd ../rust || echo 'No rust directory.'
cargo --quiet run --release test_performance
