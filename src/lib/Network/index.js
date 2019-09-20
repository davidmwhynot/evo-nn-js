const Layer = require('./Layer');

class Network {
	constructor({ layers, learningRate }) {
		this.Layers = [];
		this.learningRate = learningRate;

		for (const layer of layers) {
			this.Layers.push(new Layer(layer.numNodes));
		}

		this.connectLayers();
	}

	setBiases(biases) {
		if (this.Layers.length - 1 !== biases.length) {
			throw new Error("Invalid input biases: dimensions must match network's.");
		} else {
			for (let i = 1; i < this.Layers.length; ++i) {
				this.Layers[i].setBiases(biases[i - 1]);
			}
		}
	}

	setWeights(weights) {
		if (this.Layers.length - 1 !== weights.length) {
			throw new Error(
				"Invalid input weights: dimensions must match network's."
			);
		} else {
			for (let i = 1; i < this.Layers.length; ++i) {
				this.Layers[i].setWeights(weights[i - 1]);
			}
		}
	}

	getWeights() {
		const output = [];
		for (let i = 1; i < this.Layers.length; ++i) {
			output.push(this.Layers[i].getWeights());
		}
		return output;
	}

	getBiases() {
		const output = [];
		for (let i = 1; i < this.Layers.length; ++i) {
			output.push(this.Layers[i].getBiases());
		}
		return output;
	}

	connectLayers() {
		for (let i = 1; i < this.Layers.length; ++i) {
			this.Layers[i].connectNodes(this.Layers[i - 1]);
		}
	}

	setInputs(values) {
		for (let i = 0; i < values.length; ++i) {
			this.Layers[0].Nodes[i].value = values[i];
		}
	}

	calculate() {
		for (let i = 1; i < this.Layers.length; ++i) {
			this.Layers[i].calculate();
		}
	}

	getOutputs() {
		let output = [];

		for (const node of this.Layers[this.Layers.length - 1].Nodes) {
			output.push(node.value);
		}

		return output;
	}

	inspect() {
		for (const layer of this.Layers) {
			for (const node of layer.Nodes) {
				console.log('node\tval: ' + node.value + ' bias: ' + node.bias);
				for (const connection of node.Connections) {
					console.log('\t', connection.weight);
				}
				console.log();
			}
		}
	}
}

module.exports = Network;
