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

pub struct Layer {
    pub size: usize,
    pub activation_function: Box<dyn ActivationFunction>,
}

#[derive(Clone, Debug)]
pub struct LinearCache {
    a_prev: Array2<f32>,
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
        self.params = parameters_update(&self.layers, &self.params, gradients, learning_rate);
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

    pub fn train_model(
        &mut self,
        x_train_data: &Array2<f32>,
        y_train_data: &Array2<f32>,
        iterations: usize,
        learning_rate: f32,
    ) {
        // set decay rate
        // let decay_rate: f32 = 0.99;

        for i in 0..iterations {
            let (al, caches) = self.forward_propagate(x_train_data);
            let grads = self.backward_propagate(&al, y_train_data, caches);
            self.update_params(&grads.clone(), learning_rate);

            // decay learning rate
            // self.learning_rate *= decay_rate.powf(i as f32);

            if i % 5 == 0 {
                let (corrects, score) = self.score(&al, y_train_data);
                println!(
                    "Epoch: {}/{} - Score: {} (corrects: {}) - Learning rate: {}",
                    i, iterations, score, corrects, self.learning_rate
                );
            }
        }
    }

    pub fn predict(&self, x_test_data: &Array2<f32>) -> Array2<f32> {
        let (al, _) = self.forward_propagate(x_test_data);
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
    let mut parameters: HashMap<String, Array2<f32>> = HashMap::new();

    // random number generator
    let mut rng = rand::thread_rng();
    // let between = Uniform::from(-0.5..0.5); // random number between a fixed range

    for l in 1..layers.len() {
        // He Initialization
        let scale = (2.0 / layer_size(&layers[l - 1]) as f32).sqrt();
        let between = Uniform::from(-scale..scale);

        let weight_array_size = layer_size(&layers[l]) * layer_size(&layers[l - 1]);
        let weight_array: Vec<f32> = (0..weight_array_size)
            .map(|_| between.sample(&mut rng))
            .collect();

        let biases_array: Vec<f32> = (0..layer_size(&layers[l]))
            .map(|_| between.sample(&mut rng))
            .collect();

        let weight_matrix = Array::from_shape_vec(
            (layer_size(&layers[l]), layer_size(&layers[l - 1])),
            weight_array,
        )
        .unwrap();

        let biases_matrix =
            Array::from_shape_vec((layer_size(&layers[l]), 1), biases_array).unwrap();

        let weight_string = ["W", &l.to_string()].join("").to_string();
        let biases_string = ["b", &l.to_string()].join("").to_string();

        println!("{} - {:?}", weight_string, weight_matrix);
        println!("{} - {:?}", biases_string, biases_matrix);

        parameters.insert(weight_string, weight_matrix);
        parameters.insert(biases_string, biases_matrix);
    }

    // test code used to force weights and biases
    // let w1 = load_array_from_csv("srcw1.csv".into());
    // let w2 = load_array_from_csv("src/w2.csv".into());
    // let b1 = load_array_from_csv("src/b1.csv".into());
    // let b2 = load_array_from_csv("src/b2.csv".into());

    // parameters.insert("W1".into(), w1);
    // parameters.insert("W2".into(), w2);
    // parameters.insert("b1".into(), b1);
    // parameters.insert("b2".into(), b2);

    parameters
}

pub fn parameters_update(
    layers: &Vec<Layer>,
    params: &HashMap<String, Array2<f32>>,
    gradients: &HashMap<String, Array2<f32>>,
    learning_rate: f32,
) -> HashMap<String, Array2<f32>> {
    let mut new_params: HashMap<String, Array2<f32>> = HashMap::new();

    for l in 1..layers.len() {
        let weight_string_grad = array_index("weight_derivative", &l);
        let bias_string_grad = array_index("bias_derivative", &l);
        let weight_string = array_index("weight", &l);
        let bias_string = array_index("bias", &l);

        new_params.insert(
            weight_string.clone(),
            params[&weight_string].clone()
                - (learning_rate * gradients[&weight_string_grad].clone()),
        );
        new_params.insert(
            bias_string.clone(),
            params[&bias_string].clone() - (learning_rate * gradients[&bias_string_grad].clone()),
        );
    }

    new_params
}

// forward propagation
pub fn forward_propagate(
    layers: &Vec<Layer>,
    params: &HashMap<String, Array2<f32>>,
    inputs: &Array2<f32>,
) -> (Array2<f32>, HashMap<String, (LinearCache, ActivationCache)>) {
    let mut a_prev = inputs.clone();
    let mut caches: HashMap<String, (LinearCache, ActivationCache)> = HashMap::new();

    // we start from index one since we skip the input layer
    for l in 1..layers.len() {
        let current_layer = &layers[l];
        let weights = &params[&array_index("weight", &l)];
        let biases = &params[&array_index("bias", &l)];

        let (a, cache_current) =
            linear_forward_activation(&a_prev, weights, biases, &current_layer.activation_function)
                .unwrap();

        caches.insert(l.to_string(), cache_current);
        a_prev = a;
    }

    // a_prev will be the activation of the output layer
    (a_prev, caches)
}

pub fn linear_forward_activation(
    a_prev: &Array2<f32>,
    w: &Array2<f32>,
    b: &Array2<f32>,
    activation_fun: &Box<dyn ActivationFunction>,
) -> Result<(Array2<f32>, (LinearCache, ActivationCache)), String> {
    let z = w.dot(a_prev) + b;
    let a = activation_fun.activate(z.clone());

    let linear_cache = LinearCache {
        a_prev: a_prev.clone(),
        w: w.clone(),
        b: b.clone(),
    };

    Ok((a, (linear_cache, ActivationCache { z })))
}

// backward propagation
pub fn backward_propagate(
    layers: &Vec<Layer>,
    outputs: &Array2<f32>,
    labels: &Array2<f32>,
    caches: HashMap<String, (LinearCache, ActivationCache)>,
) -> HashMap<String, Array2<f32>> {
    let mut gradients = HashMap::new();
    let mut dz_prev = outputs.clone();
    // let mut dz_prev = -(labels / outputs - (1.0 - labels) / (1.0 - outputs));

    for l in (1..layers.len()).rev() {
        let current_cache = caches[&l.to_string()].clone();

        let (dz, dw, db) = linear_backward_activation(
            &dz_prev,
            current_cache,
            labels,
            &layers[l].activation_function,
        );

        gradients.insert(array_index("weight_derivative", &l), dw);
        gradients.insert(array_index("bias_derivative", &l), db);
        gradients.insert(array_index("activation_derivative", &l), dz.clone());

        dz_prev = dz;
    }

    gradients
}

pub fn linear_backward_activation(
    dz_prev: &Array2<f32>,
    cache: (LinearCache, ActivationCache),
    labels: &Array2<f32>,
    activation_fun: &Box<dyn ActivationFunction>,
) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let (linear_cache, activation_cache) = cache;
    let (a_prev, w, _b) = (linear_cache.a_prev, linear_cache.w, linear_cache.b);

    let dz: Array2<f32> = activation_fun.derive(dz_prev.clone(), activation_cache.z, labels);

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
