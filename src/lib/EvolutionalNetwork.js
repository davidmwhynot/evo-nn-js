const Network = require('./Network');

class EvolutionalNetwork extends Network {
	constructor({ layers, learningRate, numChildNetworks }) {
		super({ layers, learningRate });

		this.childNetworks = [];
		// this.childNetworkErrors = [];

		for (let i = 0; i < numChildNetworks; ++i) {
			this.childNetworks.push(new Network({ layers, learningRate }));
		}
	}

	// train on a batch of samples
	train(samples, batchSize) {
		return new Promise(async resolve => {
			console.log('train');
			const batches = Math.floor(samples.length / batchSize);

			for (let i = 0; i < batches; ++i) {
				process.stdout.write('.');
				const batch = [];

				for (let j = 0; j < batchSize; ++j) {
					batch.push(samples[j + i * batchSize]);
				}

				await this.runBatch(batch);
			}
			console.log('\n');
			resolve();
		});
	}

	runBatch(batch) {
		// wait for all batches to finish
		return new Promise(resolve => {
			Promise.all(
				this.childNetworks.map(network => {
					return new Promise(async resolve => {
						network.setWeights(this.getWeights());
						network.setBiases(this.getBiases());
						this.adjustNetwork(network);

						resolve(await this.calculateBatch(network, batch));
					});
				})
			).then(errors => {
				// console.log(errors);
				const min = Math.min.apply(null, errors);
				// console.log(min);
				// update the main network with the weights/biases of the top performer
				const bestChild = this.childNetworks[errors.indexOf(min)];
				this.setWeights(bestChild.getWeights());
				this.setBiases(bestChild.getBiases());
				resolve();
			});
		});
	}

	calculateBatch(network, batch) {
		return new Promise(resolve => {
			let error = 0;
			for (const sample of batch) {
				error += this.runSample(network, sample);
			}
			// reduced = errors.reduce();
			resolve(error / batch.length);
		});
	}

	runSample(network, sample) {
		// run a network and return its accuracy on a sample
		network.setInputs(sample.X);

		// compute outputs
		network.calculate();
		const outputs = network.getOutputs();

		// compute the error for the sample across all outputs
		const errors = [];

		for (let i = 0; i < outputs.length; ++i) {
			// ERROR = |TARGET - OUTPUT|
			errors.push(Math.abs(sample.y[i] - outputs[i]));
		}

		// return the cumulative error for the sample
		return errors.reduce((a, b) => a + b, 0);
	}

	adjustNetwork(network) {
		// adjust weights randomly
		for (const layer of network.Layers) {
			for (const node of layer.Nodes) {
				for (const connection of node.Connections) {
					// let before = connection.weight;
					connection.weight =
						connection.weight * ((Math.random() - 0.5) * this.learningRate) +
						(Math.random() - 0.5) * this.learningRate;
					// console.log(before - connection.weight);
					// console.log();
				}
			}
		}

		// adjust biases randomly
		for (const layer of network.Layers) {
			for (const node of layer.Nodes) {
				node.bias =
					node.bias * ((Math.random() - 0.5) * this.learningRate) +
					(Math.random() - 0.5) * this.learningRate;
			}
		}
	}
}

module.exports = EvolutionalNetwork;
