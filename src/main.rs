use itertools::zip;
fn main() {
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
}

// ---------------------------------------------
// Pg.26 Hard coding a neuron with 3 inputs and 3 weights
// fn hard_coded_neuron_ex1() -> f32 {
//     let inputs = vec![1.0, 2.0, 3.0];
//     let weights = vec![0.2, 0.8, -0.5];
//     let bias = 2.0;

//     let output = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + bias;

//     return output;
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

//     return output;
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

//     return outputs;
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

//     for (neuron_weights, neuron_bias) in zip(weights, biases) {
//         let mut neuron_output: f32 = 0.0;

//         for (n_input, weight) in zip(&inputs, neuron_weights) {
//             neuron_output += n_input * weight;
//         }
//         neuron_output += neuron_bias;

//         layer_outputs.push(neuron_output)
//     }

//     return layer_outputs;
// }
