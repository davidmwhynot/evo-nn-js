const Layer = require('./Layer');
const Node = require('./Node');
const Connection = require('./Connection');
const Matrix = require('./Matrix');

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

	feedForward(inputArray) {
		let inputs = Matrix.fromArray(inputArray);
		for (let i = 1; i < this.Layers.length; ++i) {
			const weights = new Matrix(
				this.Layers[i].Nodes.length,
				this.Layers[i - 1].Nodes.length
			);
			weights.data = this.Layers[i].getWeights();

			const biases = new Matrix(this.Layers[i].Nodes.length);
			biases.data = this.Layers[i].getBiases();

			console.time('cpu');
			// calculate values
			inputs = Matrix.multiply(weights, inputs);
			inputs.add(biases);
			inputs.map(sigmoid);
			console.timeEnd('cpu');
		}
		return inputs.toArray();
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
}

module.exports = {
	Layer,
	Node,
	Connection,
	Network,
	Matrix
};
