pub trait F64CompliantScalar: Into<f64> + From<f64> {
    fn map_f64<F>(self, func: F) -> Self
    where
        F: Fn(f64) -> f64
    {
        let x: f64 = self.into();
        Self::from(func(x))
    }
}

impl F64CompliantScalar for f64 {}
