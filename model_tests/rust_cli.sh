cd rust_cli || echo 'No rust_cli/ folder!'
cargo run --release --quiet -- "$@"
cd ../
