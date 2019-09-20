const { EvolutionalNetwork } = require('./lib');
const { shuffle } = require('./lib/utils');

// XOR problem
const xor = async ({
	trainingSamples,
	epochs,
	batchSize,
	learningRate,
	numChildNetworks
}) => {
	const network = new EvolutionalNetwork({
		layers: [{ numNodes: 2 }, { numNodes: 256 }, { numNodes: 1 }],
		learningRate,
		numChildNetworks
	});

	const REFERENCE_DATA = [
		{
			X: [0, 0],
			y: [0]
		},
		{
			X: [0, 1],
			y: [1]
		},
		{
			X: [1, 0],
			y: [1]
		},
		{
			X: [1, 1],
			y: [0]
		}
	];
	let xorTrainingData = [];

	for (let i = 0; i < trainingSamples; ++i) {
		xorTrainingData.push(REFERENCE_DATA[i % 4]);
	}

	xorTrainingData = shuffle(xorTrainingData);

	for (let i = 0; i < epochs; ++i) {
		// train
		await network.train(xorTrainingData, batchSize);

		// validate
		for (const sample of REFERENCE_DATA) {
			console.log('sample.y\t', sample.y);
			network.setInputs(sample.X);
			network.calculate();
			const outputs = network.getOutputs();
			console.log('outputs\t\t', outputs);
			console.log();
		}
	}
};

const main = async () => {
	xor({
		learningRate: 4,
		numChildNetworks: 512,
		trainingSamples: 1000,
		epochs: 10,
		batchSize: 100
	});
};

main();
