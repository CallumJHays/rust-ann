mod layer;
pub use self::layer::Layer;
use activators::{Activator, Relu};

#[derive(Debug)]
pub struct Network {
    pub layers: Vec<Layer>,
}

// Used as a public API for construction and validation of layers in a network
pub struct LayerBlueprint {
    num_nodes: usize,
    seed_bias: Option<f64>,
    seed_weights: Option<Vec<Vec<f64>>>,
    activator: Box<Activator>,
}

impl LayerBlueprint {
    pub fn new(num_nodes: usize,
               seed_bias: Option<f64>,
               seed_weights: Option<Vec<Vec<f64>>>,
               activator: Option<Box<Activator>>)
               -> Self {

        assert!(num_nodes >= 1);

        if let Some(seed_weights) = seed_weights.as_ref() {
            assert_eq!(seed_weights.len(), num_nodes)
        }

        LayerBlueprint {
            num_nodes: num_nodes,
            seed_bias: seed_bias,
            seed_weights: seed_weights,
            activator: activator.unwrap_or_else(|| Box::new(Relu)),
        }
    }
}

impl Network {
    pub fn new(mut num_inputs: usize, layer_blueprints: Vec<LayerBlueprint>) -> Self {
        Network {
            layers: layer_blueprints.into_iter()
                .map(|blueprint| {
                    for input_weights in &blueprint.seed_weights {
                        assert!(input_weights.len() > 0);
                        assert_eq!(input_weights[0].len(), num_inputs);
                    }

                    let layer = Layer::new(blueprint.num_nodes,
                                           num_inputs,
                                           blueprint.seed_bias,
                                           blueprint.seed_weights,
                                           blueprint.activator);

                    num_inputs = blueprint.num_nodes;
                    layer
                })
                .collect(),
        }
    }

    // trains the network, adjusting all weights within the network to account for the way that the error
    // after an Example propogates with the weights.
    // returns the error value BEFORE this round of training.
    pub fn train(&mut self,
                 training_inputs: &[f64],
                 training_outputs: &[f64],
                 learning_rate: f64)
                 -> f64 {

        assert_eq!(training_inputs.len(), self.layers[0].weights[0].len());
        assert_eq!(training_outputs.len(),
                   self.layers.last().unwrap().weights.len());

        // feed-forward
        // calculate the outputs of each layer in order and find our final answer
        let mut network_outputs = vec![Vec::from(training_inputs)];
        for layer in &self.layers {
            let next_output = layer.calculate_output(network_outputs.last().unwrap());
            network_outputs.push(next_output);
        }

        // back propagation
        // loop through the layers backwards and propagate the error throughout
        let mut layers_backwards: Vec<&mut Layer> = self.layers.iter_mut().collect();
        layers_backwards.reverse();

        // Necessary to get around borrow rules
        let num_layers = layers_backwards.len();

        // keeps track of the error the current layer being adjusted is experiencing.
        let mut layer_error: Vec<f64> = network_outputs.last()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, output)| output - training_outputs[i])
            .collect();

        // net error for BEFORE the training (to return)
        let net_error = layer_error.iter().map(|x| 0.5 * x.powi(2)).sum();

        for (layer_i, layer) in layers_backwards.iter_mut().enumerate() {
            let layer_i = num_layers - layer_i;
            // to build up the next layer's error
            let mut next_layer_error: Vec<f64> = layer.weights[0].iter().map(|_| 0.).collect();

            for (out_i, input_weights) in layer.weights.iter_mut().enumerate() {
                for (in_i, weight) in input_weights.iter_mut().enumerate() {
                    // do the chain rule dance!
                    let input_err = layer_error[out_i] *
                                    layer.activator.derived(network_outputs[layer_i][out_i]);
                    next_layer_error[in_i] += input_err * *weight;

                    *weight -= learning_rate * input_err * network_outputs[layer_i - 1][in_i];
                }
            }

            // TODO: Adjust layer bias as well for faster training.

            layer_error = next_layer_error;
        }

        net_error
    }
}
