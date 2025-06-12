const double learningRate = 0.3;
const int epochs = 4000;

var random = new Random();
var neuralNetwork = new NeuralNetwork.NeuralNetwork(2, 1, 2, 1, learningRate);

var trainingData = new List<(double[] input, double[] expected)>
{
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0])
};

for (var i = 0; i < epochs; i++)
{
    trainingData.ForEach(trainingSample =>
    {
        var totalError = neuralNetwork.Train(trainingSample.input, trainingSample.expected);
    
        Console.WriteLine(i + ": " + totalError);
    });
}

trainingData.ForEach(kvp =>
{
    Console.WriteLine("Value: " + neuralNetwork.GetCurrentOutputs(kvp.input)[0] + " | Expected: " + kvp.expected[0]);
});
