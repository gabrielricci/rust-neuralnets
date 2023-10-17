pub mod activation;
pub mod datasets;

use crate::neuralnet::activation::*;

use ndarray::prelude::*;
use rand::distributions::Uniform;
use rand::prelude::*;
use std::collections::HashMap;

trait Log {
    fn log(&self) -> Array2<f32>;
}

impl Log for Array2<f32> {
    fn log(&self) -> Array2<f32> {
        self.mapv(|x| x.log(std::f32::consts::E))
    }
}

pub struct DeepNeuralNetwork {
    pub layers: Vec<Layer>,
    pub learning_rate: f32,
    pub params: HashMap<String, Array2<f32>>,
}

// Can't use box and derive from Clone since Clone trait requires object to be sized, thus can't use Box
// #[derive(Clone, Debug)]
pub struct Layer {
    pub size: usize,
    pub activation_function: Box<dyn ActivationFunction>,
}

#[derive(Clone, Debug)]
pub struct LinearCache {
    a: Array2<f32>,
    w: Array2<f32>,
    b: Array2<f32>,
}

#[derive(Clone, Debug)]
pub struct ActivationCache {
    z: Array2<f32>,
}

impl DeepNeuralNetwork {
    pub fn initialize_params(&mut self) {
        self.params = parameters_initialize(&self.layers);
    }

    pub fn update_params(&mut self, gradients: &HashMap<String, Array2<f32>>, learning_rate: f32) {
        self.params =
            parameters_update(&self.layers, self.params.clone(), gradients, learning_rate);
    }

    pub fn forward_propagate(
        &self,
        x: &Array2<f32>,
    ) -> (Array2<f32>, HashMap<String, (LinearCache, ActivationCache)>) {
        forward_propagate(&self.layers, &self.params, x)
    }

    pub fn backward_propagate(
        &self,
        al: &Array2<f32>,
        labels: &Array2<f32>,
        caches: HashMap<String, (LinearCache, ActivationCache)>,
    ) -> HashMap<String, Array2<f32>> {
        backward_propagate(&self.layers, al, labels, caches)
    }

    // START OF TEST FUNCTIONS
    pub fn cost(&self, al: &Array2<f32>, y: &Array2<f32>) -> f32 {
        let m = y.shape()[1] as f32;

        // Clip AL to avoid log(0) and log(1)
        // let al_clip = al.clone();
        // let al_clip = al.mapv(|x| x.max(1e-15).min(1.0 - 1e-15));

        //let yt = &y.clone().reversed_axes();
        // let cost = (-1.0 / m) * (yt * &al_clip.log() + (1.0 - yt) * &(1.0 - &al_clip).log());
        let cost = -(1.0 / m)
            * (y.dot(&al.clone().reversed_axes().log())
                + (1.0 - y).dot(&(1.0 - al).reversed_axes().log()));

        cost.sum()
    }

    pub fn train_model(
        &mut self,
        x_train_data: &Array2<f32>,
        y_train_data: &Array2<f32>,
        iterations: usize,
        learning_rate: f32,
    ) {
        let mut costs: Vec<f32> = vec![];

        for i in 0..iterations {
            let (al, caches) = self.forward_propagate(&x_train_data);
            let cost = self.cost(&al, &y_train_data);
            let grads = self.backward_propagate(&al, &y_train_data, caches);

            self.update_params(&grads.clone(), learning_rate);

            if i % 5 == 0 {
                costs.append(&mut vec![cost]);
                let (corrects, score) = self.score(&al, y_train_data);
                println!(
                    "Epoch: {}/{} - Score: {} (corrects: {})",
                    i, iterations, score, corrects
                );
            }
        }
    }

    pub fn predict(&self, x_test_data: &Array2<f32>) -> Array2<f32> {
        let (al, _) = self.forward_propagate(x_test_data);
        //let y_hat = al.map(|x| (x > &0.5) as i32 as f32);
        //y_hat
        al
    }

    pub fn score(&self, predictions: &Array2<f32>, y: &Array2<f32>) -> (usize, f32) {
        let num_samples = predictions.shape()[1];
        let mut corrects = 0;

        for col in 0..num_samples {
            let mut expected = -1.0;
            let mut predicted = -1.0;
            let mut predicted_score = -1.0;

            for row in 0..10 {
                if predictions[[row, col]] > predicted_score {
                    predicted = row as f32;
                    predicted_score = predictions[[row, col]];
                }
                if y[[row, col]] == 1.0 {
                    expected = row as f32;
                }
            }

            if expected == predicted {
                corrects += 1;
            }
        }

        (corrects, (corrects as f32 / num_samples as f32))
    }
}

// parameter handling
pub fn parameters_initialize(layers: &Vec<Layer>) -> HashMap<String, Array2<f32>> {
    // random number generator
    // let between = Uniform::from(-0.5..0.5); // random number between -1 and 1
    let mut rng = rand::thread_rng(); // random number generator

    let mut parameters: HashMap<String, Array2<f32>> = HashMap::new();

    for l in 1..layers.len() {
        // He Initialization
        let scale = (2.0 / layer_size(&layers[l - 1]) as f32).sqrt();
        let between = Uniform::from(-scale..scale);

        let weight_array_size = layer_size(&layers[l]) * layer_size(&layers[l - 1]);
        let weight_array: Vec<f32> = (0..weight_array_size)
            .map(|_| between.sample(&mut rng))
            .collect();

        let biases_array: Vec<f32> = (0..layer_size(&layers[l])).map(|_| 0.0).collect();

        let weight_matrix = Array::from_shape_vec(
            (layer_size(&layers[l]), layer_size(&layers[l - 1])),
            weight_array,
        )
        .unwrap();

        let biases_matrix =
            Array::from_shape_vec((layer_size(&layers[l]), 1), biases_array).unwrap();

        let weight_string = ["W", &l.to_string()].join("").to_string();
        let biases_string = ["b", &l.to_string()].join("").to_string();

        parameters.insert(weight_string, weight_matrix);
        parameters.insert(biases_string, biases_matrix);
    }

    parameters
}

pub fn parameters_update(
    layers: &Vec<Layer>,
    mut params: HashMap<String, Array2<f32>>,
    gradients: &HashMap<String, Array2<f32>>,
    learning_rate: f32,
) -> HashMap<String, Array2<f32>> {
    for l in 1..layers.len() {
        let weight_string_grad = array_index("weight_derivative", &l);
        let bias_string_grad = array_index("bias_derivative", &l);
        let weight_string = array_index("weight", &l);
        let bias_string = array_index("bias", &l);

        *params.get_mut(&weight_string).unwrap() = params[&weight_string].clone()
            - (learning_rate * gradients[&weight_string_grad].clone());
        *params.get_mut(&bias_string).unwrap() =
            params[&bias_string].clone() - (learning_rate * gradients[&bias_string_grad].clone());
    }

    params
}

// forward propagation
pub fn forward_propagate(
    layers: &Vec<Layer>,
    params: &HashMap<String, Array2<f32>>,
    inputs: &Array2<f32>,
) -> (Array2<f32>, HashMap<String, (LinearCache, ActivationCache)>) {
    let mut al = inputs.clone();
    let mut caches: HashMap<String, (LinearCache, ActivationCache)> = HashMap::new();

    // we start from index one since we skip the input layer
    for l in 1..layers.len() {
        let current_layer = &layers[l];
        let weights = &params[&array_index("weight", &l)];
        let biases = &params[&array_index("bias", &l)];

        let (a, cache_current) =
            linear_forward_activation(&al, weights, biases, &current_layer.activation_function)
                .unwrap();

        caches.insert(l.to_string(), cache_current);
        al = a;
    }

    (al, caches)
}

pub fn linear_forward_activation(
    a: &Array2<f32>,
    w: &Array2<f32>,
    b: &Array2<f32>,
    activation_fun: &Box<dyn ActivationFunction>,
) -> Result<(Array2<f32>, (LinearCache, ActivationCache)), String> {
    let (z, linear_cache) = linear_forward(a, w, b);
    let a_next = activation_fun.activate(z.clone());
    Ok((a_next, (linear_cache, ActivationCache { z })))
}

pub fn linear_forward(
    a: &Array2<f32>,
    w: &Array2<f32>,
    b: &Array2<f32>,
) -> (Array2<f32>, LinearCache) {
    let z = w.dot(a) + b;
    let cache = LinearCache {
        a: a.clone(),
        w: w.clone(),
        b: b.clone(),
    };

    (z, cache)
}

// backward propagation
pub fn backward_propagate(
    layers: &Vec<Layer>,
    outputs: &Array2<f32>,
    labels: &Array2<f32>,
    caches: HashMap<String, (LinearCache, ActivationCache)>,
) -> HashMap<String, Array2<f32>> {
    let mut gradients = HashMap::new();

    // calculate the derivative of the activation of the layer (output in this case)
    let mut dal = -(labels / outputs - (1.0 - labels) / (1.0 - outputs));

    // again we skip the first layer, which is the input one
    for l in (1..layers.len()).rev() {
        let current_cache = caches[&l.to_string()].clone();
        let (da_prev, dw, db) =
            linear_backward_activation(&dal, current_cache, labels, &layers[l].activation_function);

        gradients.insert(array_index("weight_derivative", &l), dw);
        gradients.insert(array_index("bias_derivative", &l), db);
        gradients.insert(array_index("activation_derivative", &l), da_prev.clone());

        dal = da_prev;
    }

    gradients
}

pub fn linear_backward_activation(
    da: &Array2<f32>,
    cache: (LinearCache, ActivationCache),
    _labels: &Array2<f32>,
    activation_fun: &Box<dyn ActivationFunction>,
) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let (linear_cache, activation_cache) = cache;
    let dz: Array2<f32> = activation_fun.derive(da.clone(), activation_cache.z);

    let (a_prev, w, _b) = (linear_cache.a, linear_cache.w, linear_cache.b);
    let m = a_prev.shape()[1] as f32;
    let dw = (1.0 / m) * (dz.dot(&a_prev.reversed_axes()));
    let db_vec = ((1.0 / m) * dz.sum_axis(Axis(1))).to_vec();
    let db = Array2::from_shape_vec((db_vec.len(), 1), db_vec).unwrap();
    let da_prev = w.reversed_axes().dot(&dz);

    (da_prev, dw, db)
}

// Function to perform one-hot encoding
pub fn one_hot_encode(labels: &Array2<f32>, num_classes: usize) -> Array2<f32> {
    let num_samples = labels.len();
    let mut one_hot_matrix = Array2::zeros((num_classes, num_samples));

    for (col, &label) in labels.iter().enumerate() {
        one_hot_matrix[[label as usize, col]] = 1.0;
    }

    one_hot_matrix
}

// helpers
fn array_index(t: &str, index: &usize) -> String {
    let prefix: &str = match t {
        "weight" => "W",
        "bias" => "b",
        "weight_derivative" => "dW",
        "bias_derivative" => "db",
        "activation_derivative" => "dA",
        _ => panic!("Unsupported array index type"),
    };

    [prefix, &index.to_string()].join("")
}

fn layer_size(layer: &Layer) -> usize {
    layer.size
}
