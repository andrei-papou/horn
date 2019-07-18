use ndarray::{Array, Array3, ArrayBase, Data, Ix3};
use num_traits::Zero;

type Pad3 = (usize, usize, usize);

pub(super) fn pad_array3<A, S>(arr: &ArrayBase<S, Ix3>, pads: &Pad3) -> Array<A, Ix3>
where
    A: Zero + Clone,
    S: Data<Elem = A>,
{
    let shape = arr.shape();
    let new_shape = (
        shape[0] + pads.0 * 2,
        shape[1] + pads.1 * 2,
        shape[2] + pads.2 * 2,
    );
    let mut new_arr = Array3::<A>::from_elem(new_shape, A::zero());
    arr.view().indexed_iter().for_each(|(idx, el)| {
        let idx = (idx.0 + pads.0, idx.1 + pads.1, idx.2 + pads.2);
        new_arr[idx] = el.clone();
    });
    new_arr
}
