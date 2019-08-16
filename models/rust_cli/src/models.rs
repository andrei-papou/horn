use horn::{model_evaluation, Backend, Container, HResult, Tensor};
use std::marker::PhantomData;

pub trait TestModel
where
    Self::Backend: Backend,
    Self::X: Tensor<<Self::Backend as Backend>::Scalar, <Self::Backend as Backend>::CommonRepr>,
    Self::Y: Tensor<<Self::Backend as Backend>::Scalar, <Self::Backend as Backend>::CommonRepr>,
{
    const NAME: &'static str;

    type Backend;
    type X;
    type Y;

    fn get_accuracy(ys: &Self::Y, ys_hat: &Self::Y) -> HResult<<Self::Y as Container>::Elem>;

    fn get_model_file_path() -> String {
        format!("../artifacts/{}.model", Self::NAME)
    }

    fn get_xs_file_path() -> String {
        format!("../artifacts/{}.x.data", Self::NAME)
    }

    fn get_ys_file_path() -> String {
        format!("../artifacts/{}.y.data", Self::NAME)
    }
}

pub struct IrisModel<B>(PhantomData<B>)
where
    B: Backend;

impl<B: Backend> TestModel for IrisModel<B> {
    const NAME: &'static str = "iris";

    type Backend = B;
    type X = B::Tensor2D;
    type Y = B::Tensor2D;

    fn get_accuracy(ys: &Self::Y, ys_hat: &Self::Y) -> HResult<<Self::Y as Container>::Elem> {
        model_evaluation::dim2::accuracy(ys, ys_hat)
    }
}

pub struct MnistMLPModel<B: Backend>(PhantomData<B>)
where
    B: Backend;

impl<B: Backend> TestModel for MnistMLPModel<B> {
    const NAME: &'static str = "mnist_mlp";

    type Backend = B;
    type X = B::Tensor2D;
    type Y = B::Tensor2D;

    fn get_accuracy(ys: &Self::Y, ys_hat: &Self::Y) -> HResult<<Self::Y as Container>::Elem> {
        model_evaluation::dim2::accuracy(ys, ys_hat)
    }
}
