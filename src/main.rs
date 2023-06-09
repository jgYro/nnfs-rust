pub mod neural;
use neural::neural::{ActivationReLu, ActivationSoftmax, DenseLayer};

use serde_json;
use std::fs;
fn main() {
    use serde::{Deserialize, Serialize};
    #[derive(Serialize, Deserialize, Debug)]
    pub struct NNFS {
        data: Vec<Vec<f32>>,
    }

    #[derive(Serialize, Deserialize, Debug)]
    pub struct NnFsY {
        data: Vec<f32>,
    }

    let path = "./nnfs_data.json";
    let data = fs::read_to_string(path).expect("Unable to read file");
    let dataset: NNFS = serde_json::from_str(&data).unwrap();

    let y_path = "./nnfs_y_data.json";
    let y_data = fs::read_to_string(y_path).expect("Unable to read file");
    let y_dataset: NnFsY = serde_json::from_str(&y_data).unwrap();

    let pre_act = DenseLayer::new(2, 3);

    let activation1 = ActivationReLu::new(pre_act.forward(dataset.data));
    let dense2 = DenseLayer::new(3, 3);

    let softmax = ActivationSoftmax::new(dense2.forward(activation1.output));

    let predictions = softmax.output.clone();

    println!(
        "This is the accuracy: {:?}",
        neural::neural::accuracy(neural::neural::argmax(&predictions, 1), y_dataset.data)
    );
}
