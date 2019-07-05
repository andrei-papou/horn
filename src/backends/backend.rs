pub type ShapeVec = Vec<usize>;
pub type TensorOpResult<T> = Result<T, String>;

pub trait Shape {
    fn shape(&self) -> ShapeVec;
}

pub trait TensorAdd<Rhs = Self> {
    type Output;

    fn tensor_add(&self, rhs: &Rhs) -> TensorOpResult<Self::Output>;
}

pub trait TensorNeg {
    type Output;

    fn tensor_neg(&self) -> TensorOpResult<Self::Output>;
}

pub trait TensorSub<Rhs = Self> : TensorAdd<Rhs>
where
    Rhs: TensorNeg<Output = Rhs>
{
    fn tensor_sub(&self, rhs: &Rhs) -> TensorOpResult<<Self as TensorAdd<Rhs>>::Output> {
        self.tensor_add(&rhs.tensor_neg()?)
    }
}

pub trait TensorMul<Rhs = Self> {
    type Output;

    fn tensor_mul(&self, rhs: &Rhs) -> TensorOpResult<Self::Output>;
}

pub trait TensorElemInv {
    type Output;

    fn tensor_elem_inv(&self) -> TensorOpResult<Self::Output>;
}

pub trait TensorDiv<Rhs = Self> {
    type Output;

    fn tensor_div(&self, rhs: &Rhs) -> TensorOpResult<Self::Output>;
}

pub trait Tensor:
    TensorAdd<Self, Output = Self> +
    TensorSub<Self, Output = Self> +
    TensorMul<Self, Output = Self> +
    TensorDiv<Self, Output = Self> +
    TensorElemInv<Output = Self> +
    TensorNeg<Output = Self> +
    Exp +
    Container +
    Shape
where
    Self: Sized
{}

pub trait Dot<Rhs> : Shape
where
    Rhs: Shape
{
    type Output;

    fn dot(&self, rhs: &Rhs) -> TensorOpResult<Self::Output>;
}

pub trait Exp {
    type Output;

    fn exp(&self) -> Self::Output;
}

pub trait Container {
    type Elem;

    fn same_from_scalar(&self, x: Self::Elem) -> Self;
}

pub trait Broadcast<To>
where
    Self: Sized
{
    fn broadcast(&self, to: &To) -> TensorOpResult<To>;
}

pub trait ReduceSum {
    type Output;

    fn reduce_sum(&self, axis: usize) -> TensorOpResult<Self::Output>;
}

pub trait ReduceMean {
    type Output;

    fn reduce_mean(&self, axis: usize) -> TensorOpResult<Self::Output>;
}

pub trait Backend {
    type Scalar;
    type CommonRepr;

    type Tensor1D;
    type Tensor2D;
    type TensorXD;
}
