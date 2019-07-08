use std::ops::Try;

use crate::serde_json::{Value, Map};

use crate::common::string_err::join;

pub struct JsonParser<'a> {
    err_path: &'a str,
}

impl<'a> JsonParser<'a> {

    pub fn new(err_path: &'a str) -> JsonParser<'a> {
        JsonParser { err_path }
    }

    pub fn get_map_key<'b>(&self, map: &'b Map<String, Value>, key: &'b str) -> Result<&'b Value, String> {
        map.get(key).into_result()
            .map_err(|e| join(format!("\"{}\" is required.", key), self.err_path))
    }

    pub fn unwrap_u64_key(&self, map: &Map<String, Value>, key: &str) -> Result<u64, String> {
        self.unwrap_u64(self.get_map_key(map, key)?, key)
    }

    pub fn unwrap_f64_key(&self, map: &Map<String, Value>, key: &str) -> Result<f64, String> {
        self.unwrap_f64(self.get_map_key(map, key)?, key)
    }

    pub fn unwrap_string_key<'b>(&self, map: &'b Map<String, Value>, key: &'b str) -> Result<&'b str, String> {
        self.unwrap_string(self.get_map_key(map, key)?, key)
    }

    pub fn unwrap_array_key<'b>(&self, map: &'b Map<String, Value>, key: &'b str) -> Result<&'b Vec<Value>, String> {
        self.unwrap_array(self.get_map_key(map, key)?, key)
    }

    pub fn unwrap_object_key<'b>(&self, map: &'b Map<String, Value>, key: &'b str) -> Result<&'b Map<String, Value>, String> {
        self.unwrap_object(self.get_map_key(map, key)?, key)
    }

    fn get_null_error(&self, key: &str) -> String {
        join(format!("Field \"{}\" should not be null.", key), self.err_path)
    }

    pub fn ensure_object<'b>(&self, v: &Value, name: &str) -> Result<(), String> {
        match v.as_object() {
            Some(_) => Ok(()),
            None => Err(join(format!("\"{}\" should be object", name), self.err_path)),
        }
    }

    pub fn unwrap_f64(&self, v: &Value, name: &str) -> Result<f64, String> {
        match v.as_f64() {
            Some(n) => Ok(n),
            None => Err(join(format!("\"{}\" should be number (f64)", name), self.err_path)),
        }
    }

    pub fn unwrap_u64(&self, v: &Value, name: &str) -> Result<u64, String> {
        match v.as_u64() {
            Some(n) => Ok(n),
            None => Err(join(format!("\"{}\" should be number (u64)", name), self.err_path)),
        }
    }

    pub fn unwrap_string<'b>(&self, v: &'b Value, name: &'b str) -> Result<&'b str, String> {
        match v.as_str() {
            Some(s) => Ok(s),
            None => Err(join(format!("\"{}\" should be string", name), self.err_path)),
        }
    }

    pub fn unwrap_array<'b>(&self, v: &'b Value, name: &'b str) -> Result<&'b Vec<Value>, String> {
        match v.as_array() {
            Some(a) => Ok(a),
            None => Err(join(format!("\"{}\" should be array", name), self.err_path)),
        }
    }

    pub fn unwrap_object<'b>(&self, v: &'b Value, name: &'b str) -> Result<&'b Map<String, Value>, String> {
        match v.as_object() {
            Some(o) => Ok(o),
            None => Err(join(format!("\"{}\" should be object", name), self.err_path)),
        }
    }

    pub fn get_u64(&self, v: & Value, key: & str) -> Result<u64, String> {
        match v[key].as_u64() {
            Some(n) => Ok(n),
            None => Err(join(format!("Field \"{}\" should be number (u64)", key), self.err_path)),
        }
    }

    pub fn get_f64(&self, v: & Value, key: & str) -> Result<f64, String> {
        match v[key].as_f64() {
            Some(n) => Ok(n),
            None => Err(join(format!("Field \"{}\" should be number (f64)", key), self.err_path)),
        }
    }

    pub fn get_object<'b>(&self, v: &'b Value, key: &'b str) -> Result<&'b Map<String, Value>, String> {
        match v[key].as_object() {
            Some(o) => Ok(o),
            None => Err(join(format!("Field \"{}\" should be object", key), self.err_path)),
        }
    }

    pub fn get_string<'b>(&self, v: &'b Value, key: &'b str) -> Result<&'b str, String> {
        match v[key].as_str() {
            Some(s) => Ok(s),
            None => Err(join(format!("Field \"{}\" should be string", key), self.err_path)),
        }
    }

    pub fn get_array<'b>(&self, v: &'b Value, key: &'b str) -> Result<&'b Vec<Value>, String> {
        match v[key].as_array() {
            Some(a) => Ok(a),
            None => Err(join(format!("Field \"{}\" should be array", key), self.err_path)),
        }
    }
}
