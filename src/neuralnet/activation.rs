use ndarray::prelude::*;
use std::f32::consts::E;

pub trait ActivationFunction {
    fn activate(&self, z: Array2<f32>) -> Array2<f32>;
    fn derive(&self, da: Array2<f32>, z: Array2<f32>, labels: &Array2<f32>) -> Array2<f32>;
}

pub struct Sigmoid {}
pub struct ReLU {}
pub struct Softmax {}

impl ActivationFunction for Sigmoid {
    fn activate(&self, z: Array2<f32>) -> Array2<f32> {
        z.mapv(|x| sigmoid(&x))
    }

    fn derive(&self, da: Array2<f32>, z: Array2<f32>, _labels: &Array2<f32>) -> Array2<f32> {
        da * z.mapv(|x| sigmoid_derivative(&x))
    }
}

impl ActivationFunction for ReLU {
    fn activate(&self, z: Array2<f32>) -> Array2<f32> {
        z.mapv(|x| relu(&x))
    }

    fn derive(&self, da: Array2<f32>, z: Array2<f32>, _labels: &Array2<f32>) -> Array2<f32> {
        da * z.mapv(|x| relu_derivative(&x))
    }
}

impl ActivationFunction for Softmax {
    fn activate(&self, z: Array2<f32>) -> Array2<f32> {
        let exps = z.mapv(|x| x.exp());
        let sum_exps = exps.sum_axis(Axis(0));

        exps / sum_exps.insert_axis(Axis(0))
    }

    fn derive(&self, da: Array2<f32>, _z: Array2<f32>, labels: &Array2<f32>) -> Array2<f32> {
        da - labels
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
