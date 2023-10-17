use ndarray::prelude::*;
use std::f32::consts::E;

pub trait ActivationFunction {
    fn activate(&self, z: Array2<f32>) -> Array2<f32>;
    fn derive(&self, da: Array2<f32>, z: Array2<f32>) -> Array2<f32>;
}

pub struct Sigmoid {}
pub struct ReLU {}
pub struct Softmax {}

impl ActivationFunction for Sigmoid {
    fn activate(&self, z: Array2<f32>) -> Array2<f32> {
        z.mapv(|x| sigmoid(&x))
    }

    fn derive(&self, da: Array2<f32>, z: Array2<f32>) -> Array2<f32> {
        da * z.mapv(|x| sigmoid_derivative(&x))
    }
}

impl ActivationFunction for ReLU {
    fn activate(&self, z: Array2<f32>) -> Array2<f32> {
        z.mapv(|x| relu(&x))
    }

    fn derive(&self, da: Array2<f32>, z: Array2<f32>) -> Array2<f32> {
        da * z.mapv(|x| relu_derivative(&x))
    }
}

impl ActivationFunction for Softmax {
    fn activate(&self, z: Array2<f32>) -> Array2<f32> {
        z.mapv(|x| softmax(&x))
    }

    fn derive(&self, _da: Array2<f32>, z: Array2<f32>) -> Array2<f32> {
        softmax_derivative(z)
    }
}

// sigmoid functions
fn sigmoid(z: &f32) -> f32 {
    1.0 / (1.0 + E.powf(-z))
}

fn sigmoid_derivative(z: &f32) -> f32 {
    sigmoid(z) * (1.0 - sigmoid(z))
}

// relu function
fn relu(z: &f32) -> f32 {
    match *z > 0.0 {
        true => *z,
        false => 0.0,
    }
}

fn relu_derivative(z: &f32) -> f32 {
    match *z > 0.0 {
        true => 1.0,
        false => 0.0,
    }
}

// softmax functions
// FIXME: This is broken, NN is not learning when using softmax as activation
fn softmax(z: &f32) -> f32 {
    E.powf(*z) / E.powf(*z).exp()
}

fn softmax_derivative(z: Array2<f32>) -> Array2<f32> {
    let exp_values = z.mapv(|x| E.powf(x));
    exp_values.clone() / &exp_values.sum_axis(Axis(1)).insert_axis(Axis(1))
}
