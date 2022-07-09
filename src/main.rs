use serde::{Deserialize, Serialize};
use serde_json;
use std::fs;
fn main() {
    #[derive(Serialize, Deserialize, Debug)]
    struct NNFS {
        data: Vec<Vec<f32>>,
    }

    let path = "./nnfs_data.json";
    let data = fs::read_to_string(path).expect("Unable to read file");
    let dataset: NNFS = serde_json::from_str(&data).unwrap();

    // Pg. 26
    // println!(
    //     "This is the first example output: {:?}",
    //     hard_coded_neuron_ex1()
    // );

    // Pg. 29
    // println!(
    //     "This is the second example output: {:?}",
    //     hard_coded_neuron_ex2()
    // )

    //Pg.31
    // println!(
    //     "This is the third example output: {:?}",
    //     hard_coded_multiple_neurons_ex1()
    // )

    //Pg.33
    // println!(
    //     "This is the fourth example output: {:?}",
    //     looping_through_neurons()
    // )

    //Pg. 38
    // println!(
    //     "This is the dot product result: {:?}",
    //     dot_product([1.0, 2.0, 3.0].to_vec(), [2.0, 3.0, 4.0].to_vec())
    // )

    //Pg. 40
    // println!(
    //     "This is the neuron with the dot product function: {:?}",
    //     neuron_with_dot_product()
    // )

    //Pg.42
    // println!(
    //     "This is a layer of neurons with the dot product function: {:?}",
    //     multiple_neurons_with_dot_product()
    // )

    //Pg.47
    // let matrix1 = vec![
    //     [0.49, 0.97, 0.53, 0.05].to_vec(),
    //     [0.33, 0.65, 0.62, 0.51].to_vec(),
    //     [1.00, 0.38, 0.61, 0.45].to_vec(),
    //     [0.74, 0.27, 0.64, 0.17].to_vec(),
    //     [0.36, 0.17, 0.96, 0.12].to_vec(),
    // ];

    // let matrix2 = vec![
    //     [0.79, 0.32, 0.68, 0.90, 0.77].to_vec(),
    //     [0.18, 0.39, 0.12, 0.93, 0.09].to_vec(),
    //     [0.87, 0.42, 0.60, 0.71, 0.12].to_vec(),
    //     [0.45, 0.55, 0.40, 0.78, 0.81].to_vec(),
    // ];

    // println!(
    //     "This is the result of the matrix multiplcation: {:?}",
    //     matrix_product(matrix1, matrix2)
    // )

    //Pg.50
    // let matrix1 = vec![
    //     [0.0, 0.1, 0.2, 0.3, 0.4].to_vec(),
    //     [0.5, 0.6, 0.7, 0.8, 0.9].to_vec(),
    //     [1.0, 1.1, 1.2, 1.3, 1.4].to_vec(),
    //     [1.5, 1.6, 1.7, 1.8, 1.9].to_vec(),
    // ];

    // let matrix2 = vec![
    //     [0.0, 0.5, 1.0, 1.5].to_vec(),
    //     [0.1, 0.6, 1.1, 1.6].to_vec(),
    //     [0.2, 0.7, 1.2, 1.7].to_vec(),
    //     [0.3, 0.8, 1.3, 1.8].to_vec(),
    //     [0.4, 0.9, 1.4, 1.9].to_vec(),
    // ];

    // matrix_transpose(matrix2);

    //Pg.58
    // println!(
    //     "This is a the layer outputs of a neuron function using matrix products and vector addition: {:?}",
    //     neuron_with_functions()
    //         );

    // Pg.61
    // println!(
    //     "This is the output of the function with the hidden layers: {:?}",
    //     neuron_with_hidden_layers()
    // )

    // Pg.69
    // let neuron_with_functions = DenseLayer {
    //     weights: vec![
    //         [0.2, 0.8, -0.5, 1.0].to_vec(),
    //         [0.5, -0.91, 0.26, -0.5].to_vec(),
    //         [-0.26, -0.27, 0.17, 0.87].to_vec(),
    //     ],
    //     biases: vec![2.0, 3.0, 0.5],
    // };

    // println!(
    //     "This is the output of the forward pass: {:#?}",
    //     neuron_with_functions.forward(vec![
    //         [1.0, 2.0, 3.0, 2.5].to_vec(),
    //         [2.0, 5.0, -1.0, 2.0].to_vec(),
    //         [-1.5, 2.7, 3.3, -0.8].to_vec(),
    //     ])
    // );

    // let singe_weight = DenseLayer {
    //     weights: vec![[0.2, 0.8, -0.5, 1.0].to_vec()],
    //     biases: vec![2.0],
    // };

    // println!(
    //     "This is the output of the forward pass: {:#?}",
    //     singe_weight.forward(vec![[1.0, 2.0, 3.0, 2.5].to_vec()])
    // )
    // }

    //Pg.71
    let pre_act = DenseLayer::new(2, 3);

    println!("Here is the output: {:#?}", pre_act.forward(dataset.data));

    // ---------------------------------------------
    // Pg.26 Hard coding a neuron with 3 inputs and 3 weights
    // fn hard_coded_neuron_ex1() -> f32 {
    //     let inputs = vec![1.0, 2.0, 3.0];
    //     let weights = vec![0.2, 0.8, -0.5];
    //     let bias = 2.0;

    //     let output = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + bias;

    //     output
    // }

    // ---------------------------------------------
    // Pg.29 Hard coding a neuron with 4 inputs and 4 weights
    // fn hard_coded_neuron_ex2() -> f32 {
    //     let inputs = vec![1.0, 2.0, 3.0, 2.5];
    //     let weights = vec![0.2, 0.8, -0.5, 1.0];
    //     let bias = 2.0;

    //     let output = inputs[0] * weights[0]
    //         + inputs[1] * weights[1]
    //         + inputs[2] * weights[2]
    //         + inputs[3] * weights[3]
    //         + bias;

    //     output
    // }

    // ---------------------------------------------
    // Pg.31 Hard coding multiple neurons in a layer
    // fn hard_coded_multiple_neurons_ex1() -> Vec<f32> {
    //     let inputs = vec![1.0, 2.0, 3.0, 2.5];

    //     let weights1 = vec![0.2, 0.8, -0.5, 1.0];
    //     let weights2 = vec![0.5, -0.91, 0.26, -0.5];
    //     let weights3 = vec![-0.26, -0.27, 0.17, 0.87];
    //     let bias1 = 2.0;
    //     let bias2 = 3.0;
    //     let bias3 = 0.5;

    //     let outputs = vec![
    //         (inputs[0] * weights1[0]
    //             + inputs[1] * weights1[1]
    //             + inputs[2] * weights1[2]
    //             + inputs[3] * weights1[3]
    //             + bias1),
    //         (inputs[0] * weights2[0]
    //             + inputs[1] * weights2[1]
    //             + inputs[2] * weights2[2]
    //             + inputs[3] * weights2[3]
    //             + bias2),
    //         (inputs[0] * weights3[0]
    //             + inputs[1] * weights3[1]
    //             + inputs[2] * weights3[2]
    //             + inputs[3] * weights3[3]
    //             + bias3),
    //     ];

    //     outputs
    // }

    // ---------------------------------------------
    // Pg.33 Implementing loops to iterate each neuron
    // TODO: Create crazy one-liner to do the loop logic below
    // fn looping_through_neurons() -> Vec<f32> {
    //     let inputs = vec![1.0, 2.0, 3.0, 2.5];

    //     let weights = vec![
    //         [0.2, 0.8, -0.5, 1.0],
    //         [0.5, -0.91, 0.26, -0.5],
    //         [-0.26, -0.27, 0.17, 0.87],
    //     ];

    //     let biases = vec![2.0, 3.0, 0.5];

    //     let mut layer_outputs: Vec<f32> = Vec::new();

    //     for (neuron_weights, neuron_bias) in std::iter::zip(weights, biases) {
    //         let mut neuron_output: f32 = 0.0;

    //         for (n_input, weight) in std::iter::zip(&inputs, neuron_weights) {
    //             neuron_output += n_input * weight;
    //         }
    //         neuron_output += neuron_bias;

    //         layer_outputs.push(neuron_output)
    //     }

    //     layer_outputs
    // }

    // Pg. 38
    // TODO: Handling 2D vector and return vector of dot products like on pg.42
    // TODO: Make a fancy one-liner
    fn dot_product(v1: Vec<f32>, v2: Vec<f32>) -> f32 {
        let mut sum: f32 = 0.0;
        for (i, j) in std::iter::zip(v1, v2) {
            sum += i * j;
        }

        sum
    }

    // Pg. 40
    // fn neuron_with_dot_product() -> f32 {
    //     let inputs = vec![1.0, 2.0, 3.0, 2.5];

    //     let weights = vec![0.2, 0.8, -0.5, 1.0];
    //     let bias = 2.0;

    //     let outputs = dot_product(inputs, weights) + bias;
    //     outputs
    // }

    // Pg. 42
    // fn multiple_neurons_with_dot_product() -> Vec<f32> {
    //     let inputs = vec![1.0, 2.0, 3.0, 2.5];

    //     let weights = vec![
    //         [0.2, 0.8, -0.5, 1.0],
    //         [0.5, -0.91, 0.26, -0.5],
    //         [-0.26, -0.27, 0.17, 0.87],
    //     ];

    //     let biases = vec![2.0, 3.0, 0.5];

    //     let mut layer_outputs: Vec<f32> = Vec::new();

    //     for (neuron_weights, bias) in std::iter::zip(weights, biases) {
    //         layer_outputs.push(dot_product(inputs.to_vec(), neuron_weights.to_vec()) + bias)
    //     }

    //     layer_outputs
    // }

    // Pg. 47
    // TODO: Make a fancy one-liner
    fn matrix_product(m1: Vec<Vec<f32>>, m2: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let mut matrix: Vec<Vec<f32>> = Vec::new();
        for i in 0..m1.len() {
            let mut temp: Vec<f32> = Vec::new();
            for j in 0..m2[0].len() {
                let mut sum = 0.0;
                for k in 0..m1[0].len() {
                    sum += m1[i][k] * m2[k][j];
                }
                temp.push(sum)
            }
            matrix.push(temp)
        }
        matrix
    }

    // // Pg. 50
    fn matrix_transpose(m1: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let mut matrix: Vec<Vec<f32>> = Vec::new();
        for i in 0..m1[0].len() {
            let mut temp: Vec<f32> = Vec::new();
            for j in &m1 {
                temp.push(j[i])
            }
            matrix.push(temp);
        }
        matrix
    }

    // // Pg. 57
    fn vector_addition(m1: Vec<Vec<f32>>, v1: Vec<f32>) -> Vec<Vec<f32>> {
        let mut matrix: Vec<Vec<f32>> = Vec::new();

        for i in 0..m1.len() {
            let mut temp: Vec<f32> = Vec::new();
            for j in 0..v1.len() {
                temp.push(m1[i][j] + v1[j])
            }
            matrix.push(temp)
        }
        matrix
    }

    // Pg. 58
    // fn neuron_with_functions() -> Vec<Vec<f32>> {
    //     let inputs = vec![
    //         [1.0, 2.0, 3.0, 2.5].to_vec(),
    //         [2.0, 5.0, -1.0, 2.0].to_vec(),
    //         [-1.5, 2.7, 3.3, -0.8].to_vec(),
    //     ];
    //     let weights = vec![
    //         [0.2, 0.8, -0.5, 1.0].to_vec(),
    //         [0.5, -0.91, 0.26, -0.5].to_vec(),
    //         [-0.26, -0.27, 0.17, 0.87].to_vec(),
    //     ];

    //     let biases = vec![2.0, 3.0, 0.5];

    //     let layer_outputs = vector_addition(matrix_product(inputs, matrix_transpose(weights)), biases);

    //     layer_outputs
    // }

    //Pg. 61
    // fn neuron_with_hidden_layers() -> Vec<Vec<f32>> {
    //     let inputs = vec![
    //         [1.0, 2.0, 3.0, 2.5].to_vec(),
    //         [2.0, 5.0, -1.0, 2.0].to_vec(),
    //         [-1.5, 2.7, 3.3, -0.8].to_vec(),
    //     ];
    //     let weights = vec![
    //         [0.2, 0.8, -0.5, 1.0].to_vec(),
    //         [0.5, -0.91, 0.26, -0.5].to_vec(),
    //         [-0.26, -0.27, 0.17, 0.87].to_vec(),
    //     ];

    //     let biases = vec![2.0, 3.0, 0.5];

    //     let weights2 = vec![
    //         [0.1, -0.14, 0.5].to_vec(),
    //         [-0.5, 0.12, -0.33].to_vec(),
    //         [-0.44, 0.73, -0.13].to_vec(),
    //     ];

    //     let biases2 = vec![-1.0, 2.0, -0.5];

    //     let layer_outputs1 = vector_addition(matrix_product(inputs, matrix_transpose(weights)), biases);
    //     let layer_outputs2 = vector_addition(
    //         matrix_product(layer_outputs1, matrix_transpose(weights2)),
    //         biases2,
    //     );

    //     layer_outputs2
    // }

    // Pg. 67
    fn create_distribution(size: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();

        let v: Vec<f32> = rand::distributions::Distribution::sample_iter(
            &rand::distributions::Standard,
            &mut rng,
        )
        .take(size)
        .collect();

        let mut distribution: Vec<f32> = Vec::new();

        for number in v {
            let rando: u8 = rand::Rng::gen(&mut rng);

            let result = if let 0 = rando % 2 {
                number
            } else {
                number * -1.0
            };
            distribution.push(result);
        }

        distribution
    }

    fn create_weights(inputs: usize, neurons: usize) -> Vec<Vec<f32>> {
        let mut weights = Vec::new();
        for _ in 0..inputs {
            weights.push(create_distribution(neurons))
        }

        weights
    }

    fn create_biases(neurons: usize) -> Vec<f32> {
        let mut biases = Vec::new();

        for _ in 0..neurons {
            biases.push(0.0);
        }
        biases
    }

    #[derive(Debug)]
    struct DenseLayer {
        weights: Vec<Vec<f32>>,
        biases: Vec<f32>,
    }

    //Pg. 69
    //This has produced the same output as two earlier examples
    impl DenseLayer {
        fn new(inputs: usize, neurons: usize) -> Self {
            return DenseLayer {
                weights: create_weights(inputs, neurons),
                biases: create_biases(neurons),
            };
        }
    }

    impl DenseLayer {
        fn forward(&self, inputs: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
            vector_addition(
                matrix_product(inputs.to_vec(), self.weights.to_vec()),
                self.biases.to_vec(),
            )
        }
    }
}
