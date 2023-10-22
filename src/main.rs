mod neuralnet;

use neuralnet::activation::*;
use neuralnet::datasets::*;
use neuralnet::*;
use std::collections::HashMap;

fn main() {
    let iterations = 1000;
    let num_classes = 10;

    let (training_data, training_labels) =
        dataframe_from_csv("src/mnist_train.csv".into()).unwrap();
    let (test_data, test_labels) = dataframe_from_csv("src/mnist_test.csv".into()).unwrap();

    let training_data_array = array_from_dataframe(&training_data) / 255.0;
    let training_labels_array = array_from_dataframe(&training_labels);
    let test_data_array = array_from_dataframe(&test_data) / 255.0;
    let test_labels_array = array_from_dataframe(&test_labels);

    let one_hot_matrix_training = one_hot_encode(&training_labels_array, num_classes);
    let one_hot_matrix_test = one_hot_encode(&test_labels_array, num_classes);

    let mut model = DeepNeuralNetwork {
        layers: vec![
            Layer {
                size: 784,
                activation_function: Box::new(ReLU {}),
            },
            Layer {
                size: 10,
                activation_function: Box::new(ReLU {}),
            },
            Layer {
                size: 10,
                activation_function: Box::new(Softmax {}),
            },
        ],
        learning_rate: 0.0075,
        params: HashMap::new(),
    };

    model.initialize_params();

    model.train_model(
        &training_data_array,
        &one_hot_matrix_training,
        iterations,
        model.learning_rate,
    );
    // write_parameters_to_json_file(&parameters, "model.json".into());

    let training_predictions = model.predict(&training_data_array);
    let (_training_corrects, training_score) =
        model.score(&training_predictions, &one_hot_matrix_training);
    println!("Training Set Accuracy: {}%", training_score);

    let test_predictions = model.predict(&test_data_array);
    let (_test_corrects, test_score) = model.score(&test_predictions, &one_hot_matrix_test);
    println!("Test Set Accuracy: {}%", test_score);
}
