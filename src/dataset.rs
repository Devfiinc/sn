use ndarray::{ArrayBase, Axis, Data, Dimension};




pub struct dataset<R, T> {
    pub x: Vec<R>,
    pub y: Vec<T>,
}