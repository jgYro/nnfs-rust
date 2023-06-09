// TODO: Handling 2D vector and return vector of dot products like on pg.42
// TODO: Make a fancy one-liner
pub fn dot_product(v1: Vec<f32>, v2: Vec<f32>) -> f32 {
    let mut sum: f32 = 0.0;
    for (i, j) in std::iter::zip(v1, v2) {
        sum += i * j;
    }

    sum
}

// TODO: Make a fancy one-liner
pub fn matrix_product(m1: Vec<Vec<f32>>, m2: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
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
pub fn matrix_transpose(m1: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
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
pub fn vector_addition(m1: Vec<Vec<f32>>, v1: Vec<f32>) -> Vec<Vec<f32>> {
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

// Pg. 67
pub fn create_distribution(size: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();

    let v: Vec<f32> =
        rand::distributions::Distribution::sample_iter(&rand::distributions::Standard, &mut rng)
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

pub fn create_weights(inputs: usize, neurons: usize) -> Vec<Vec<f32>> {
    let mut weights = Vec::new();
    for _ in 0..inputs {
        weights.push(create_distribution(neurons))
    }

    weights
}

pub fn create_biases(neurons: usize) -> Vec<f32> {
    let mut biases = Vec::new();

    for _ in 0..neurons {
        biases.push(0.0);
    }
    biases
}

#[derive(Debug)]
pub struct DenseLayer {
    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<f32>,
}

////Pg. 69
impl DenseLayer {
    pub fn new(inputs: usize, neurons: usize) -> Self {
        return DenseLayer {
            weights: create_weights(inputs, neurons),
            biases: create_biases(neurons),
        };
    }
}

impl DenseLayer {
    pub fn forward(&self, inputs: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        vector_addition(
            matrix_product(inputs.to_vec(), self.weights.to_vec()),
            self.biases.to_vec(),
        )
    }
}

////Pg. 95
#[derive(Debug)]
pub struct ActivationReLu {
    pub output: Vec<Vec<f32>>,
}

impl ActivationReLu {
    pub fn new(inputs: Vec<Vec<f32>>) -> Self {
        let mut activation: Vec<Vec<f32>> = Vec::new();

        for vector in inputs {
            let mut re_lu: Vec<f32> = Vec::new();
            for float in vector {
                if float > 0.0 {
                    re_lu.push(float)
                } else {
                    re_lu.push(0.0)
                }
            }
            activation.push(re_lu)
        }
        return ActivationReLu { output: activation };
    }
}

#[derive(Debug)]
pub struct ActivationSoftmax {
    pub output: Vec<Vec<f32>>,
}

impl ActivationSoftmax {
    //Pg. 101
    pub fn new(outputs: Vec<Vec<f32>>) -> Self {
        let mut normalize_outputs = Vec::new();
        for row in outputs {
            let e = 2.71828182846;

            let mut exp_values = Vec::new();

            for output in row {
                exp_values.push(f32::powf(e, output));
            }

            let mut norm_base = 0.0;

            for val in &exp_values {
                norm_base += val
            }

            let mut norm_values = Vec::new();

            for val in exp_values {
                norm_values.push(val / norm_base)
            }

            normalize_outputs.push(norm_values)
        }
        ActivationSoftmax {
            output: normalize_outputs,
        }
    }
}

#[derive(Debug)]
pub struct Loss {
    pub output: f32,
}

impl Loss {
    // One dimensional array, for categorical labels
    pub fn categorical_loss(targets: Vec<f32>, outputs: Vec<Vec<f32>>) -> Self {
        let mut neg_loss = 0.0;
        for (target_idx, distribution) in std::iter::zip(targets, &outputs) {
            neg_loss += f32::ln(distribution[target_idx as usize]);
        }
        return Loss {
            output: -neg_loss / outputs.len() as f32,
        };
    }

    // Two dimensional array, for one hot encoded labels.
    pub fn one_hot_loss(targets: Vec<Vec<f32>>, outputs: Vec<Vec<f32>>) -> Self {
        let mut neg_loss = 0.0;
        for (target_vec, output_vec) in std::iter::zip(&targets, &outputs) {
            for (target, output) in std::iter::zip(target_vec, output_vec) {
                if (target * output) != 0.0 {
                    neg_loss += f32::ln(target * output)
                }
            }
        }
        return Loss {
            output: -neg_loss / outputs.len() as f32,
        };
    }
}

// TODO: add min functionality
pub fn argmax(input: &Vec<Vec<f32>>, axis: usize) -> Vec<f32> {
    let mut maxes: Vec<f32> = Vec::new();
    let mut output: Vec<f32> = Vec::new();
    let data: Vec<Vec<f32>> = if axis < 1 {
        matrix_transpose(input.to_vec())
    } else {
        input.to_vec()
    };

    // Column
    for row in &data {
        let mut max: (usize, f32) = (0, 0.0);
        for (i, r) in row.into_iter().enumerate() {
            if r > &max.1 {
                max.0 = i;
                max.1 = *r
            }
        }
        maxes.push(max.1)
    }
    for (m, v) in std::iter::zip(maxes, data) {
        output.push(v.iter().position(|&x| x == m).unwrap() as f32)
    }
    output
}
// TODO: add mean and flat mean as one function, with optional parameters
pub fn mean(m1: Vec<Vec<f32>>, axis: usize) -> Vec<f32> {
    let mut output: Vec<f32> = Vec::new();

    let data: Vec<Vec<f32>> = if axis < 1 {
        matrix_transpose(m1.to_vec())
    } else {
        m1.to_vec()
    };
    // Column
    for row in &data {
        let mut temp: f32 = 0.0;
        for r in row {
            temp += r;
        }
        output.push(temp / row.len() as f32)
    }
    output
}
pub fn flat_mean(m1: Vec<Vec<f32>>) -> Vec<f32> {
    let mut output: Vec<f32> = Vec::new();

    let mut count: usize = 0;
    let mut temp: f32 = 0.0;
    for row in m1 {
        for r in row {
            temp += r;
            count += 1;
        }
    }
    output.push(temp / count as f32);
    output
}

pub fn equal_mean(v1: Vec<f32>, v2: Vec<f32>) -> Vec<f32> {
    let mut output: Vec<f32> = Vec::new();
    for (i, j) in std::iter::zip(v1, v2) {
        if i == j {
            output.push(1.0)
        } else {
            output.push(0.0)
        }
    }
    output
}

pub fn single_mean(v1: Vec<f32>) -> f32 {
    let mut count: usize = 0;
    let mut temp: f32 = 0.0;
    for v in v1 {
        temp += v;
        count += 1;
    }

    return temp / count as f32;
}

pub fn accuracy(v1: Vec<f32>, v2: Vec<f32>) -> f32 {
    return single_mean(equal_mean(v1, v2));
}
