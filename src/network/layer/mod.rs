extern crate rand;

use rand::{thread_rng, Rng};

use activators::Activator;

#[derive(Debug)]
pub struct Layer {
    num_nodes: usize,
    pub bias: f64,
    pub weights: Vec<Vec<f64>>,
    pub activator: Box<Activator>,
}

fn xavier_init(n_in: usize, n_out: usize) -> Vec<Vec<f64>> {
    let mut rng = thread_rng();
    let variance = (2f64 / ((n_in + n_out) as f64)).sqrt();

    (0..n_out)
        .map(|_| {
            (0..n_in)
                .map(|_| rng.gen_range(-variance, variance))
                .collect()
        })
        .collect()
}

impl Layer {
    pub fn new(num_nodes: usize,
               num_inputs: usize,
               seed_bias: Option<f64>,
               seed_weights: Option<Vec<Vec<f64>>>,
               activator: Box<Activator>)
               -> Self {

        let seed_weights = seed_weights.unwrap_or_else(|| xavier_init(num_inputs, num_nodes));
        assert_eq!(seed_weights.len(), num_nodes);
        for input_weights in &seed_weights {
            assert_eq!(input_weights.len(), num_inputs);
        }

        Layer {
            num_nodes: num_nodes,
            bias: seed_bias.unwrap_or_else(|| rand::thread_rng().gen()),
            weights: seed_weights,
            activator: activator,
        }
    }

    // calculates the output vector with activations
    pub fn calculate_output(&self, inputs: &[f64]) -> Vec<f64> {
        self.weights
            .iter()
            .map(|input_weights| {
                self.activator.activate(inputs.iter()
                    .enumerate()
                    .map(|(i, input)| input * input_weights[i])
                    .sum::<f64>() + self.bias)
            })
            .collect()
    }
}
