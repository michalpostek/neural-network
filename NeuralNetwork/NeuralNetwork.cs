namespace NeuralNetwork;

public class NeuralNetwork(int inputsCount, int hiddenLayersCount, int hiddenLayerSize, int outputsCount, double learningRate)
{
    private readonly Neuron[][] _hiddenLayers = NeuronFactory.InitHiddenLayers(inputsCount, hiddenLayersCount, hiddenLayerSize);
    private readonly Neuron[] _outputLayer = NeuronFactory.InitOutputLayer(hiddenLayerSize, outputsCount);
    private readonly double _learningRate = learningRate;

    public double[] GetCurrentOutputs(double[] inputs)
    {
        var previousLayerOutputs = inputs.ToArray();
        
        foreach (var hiddenLayer in _hiddenLayers)
        {
            previousLayerOutputs = hiddenLayer.Select(n => n.ActivateAndGetOutput(previousLayerOutputs)).ToArray();
        }

        return _outputLayer.Select(x => x.ActivateAndGetOutput(previousLayerOutputs)).ToArray();
    }

    public double Train(double[] inputs, double[] expectedOutputs)
    {
        var outputs = GetCurrentOutputs(inputs);
        var totalError = outputs.Select((output, index) => Math.Abs(expectedOutputs[index] - output)).Sum();

        // set deltas
        for (var i = 0; i < _outputLayer.Length; i++)
        {
            var error = expectedOutputs[i] - _outputLayer[i].Output!.Value;
            
            _outputLayer[i].SetDelta(error);
        }

        var nextLayer = _outputLayer;

        for (var i = _hiddenLayers.Length - 1; i >= 0; i--)
        {
            for (var j = 0; j < _hiddenLayers[i].Length; j++)
            {
                var error = nextLayer.Aggregate(0d, (current, next) => current + next.Delta!.Value * next.Weights[i]);
                
                _hiddenLayers[i][j].SetDelta(error);
            }
            
            nextLayer = _hiddenLayers[i];
        }
        
        // adjust weights
        var previousLayerOutputs = inputs;

        foreach (var layer in _hiddenLayers)
        {
            foreach (var neuron in layer)
            {
                neuron.Bias += _learningRate * neuron.Delta!.Value;
                
                for (var j = 0; j < previousLayerOutputs.Length; j++)
                {
                    neuron.Weights[j] += _learningRate * neuron.Delta!.Value * previousLayerOutputs[j];
                }
            }
            
            previousLayerOutputs = layer.Select(n => n.Output!.Value).ToArray();
        }

        foreach (var neuron in _outputLayer)
        {
            neuron.Bias += _learningRate * neuron.Delta!.Value;
            
            for (var j = 0; j < previousLayerOutputs.Length; j++)
            {
                neuron.Weights[j] += _learningRate * neuron.Delta!.Value * previousLayerOutputs[j];
            }
        }

        return totalError;
    }
}