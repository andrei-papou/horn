const LEVEL_SEPARATOR: &'static str = "\n from ";

pub fn join(mut err: String, path_cmp: &str) -> String {
    err.push_str(path_cmp);
    err
}

pub fn err_to_string<E: ToString>(err: E) -> String {
    err.to_string()
}
