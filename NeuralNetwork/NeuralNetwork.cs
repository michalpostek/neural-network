namespace NeuralNetwork;

public class NeuralNetwork(int inputsCount, int hiddenLayersCount, int hiddenLayerSize, int outputsCount, double learningRate)
{
    private readonly Neuron[][] _hiddenLayers = InitHiddenLayers(inputsCount, hiddenLayersCount, hiddenLayerSize);
    private readonly Neuron[] _outputLayer = InitOutputLayer(hiddenLayerSize, outputsCount);
    private readonly double _learningRate = learningRate;

    public double[] GetCurrentOutputs(double[] inputs)
    {
        var previousLayerOutputs = inputs.ToList();

        for (var i = 0; i < _hiddenLayers.Length; i++)
        {
            var hiddenLayerOutputs = new double[_hiddenLayers[i].Length];

            for (var j = 0; j < _hiddenLayers[i].Length; j++)
            {
                hiddenLayerOutputs[j] = _hiddenLayers[i][j].ActivateAndGetOutput(previousLayerOutputs.ToArray());
            }

            previousLayerOutputs = hiddenLayerOutputs.ToList();
        }

        return _outputLayer.Select(x => x.ActivateAndGetOutput(previousLayerOutputs.ToArray())).ToArray();
    }

    public double Train(double[] inputs, double[] expectedOutputs)
    {
        var outputs = GetCurrentOutputs(inputs);
        var squaredErrors = outputs.Select((output, index) => Math.Pow(expectedOutputs[index] - output, 2)).Sum();

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

        return squaredErrors;
    }

    private static Neuron[] InitOutputLayer(int hiddenLayerSize, int outputsCount)
    {
        var layer = new Neuron[outputsCount];

        for (var i = 0; i < outputsCount; i++)
        {
            layer[i] = NeuronFactory.CreateNeuron(hiddenLayerSize);
        }
        
        return layer;
    }

    private static Neuron[][] InitHiddenLayers(int inputsCount, int hiddenLayersCount, int hiddenLayerSize)
    {
        var hiddenLayers = new Neuron[hiddenLayersCount][];

        for (var i = 0; i < hiddenLayersCount; i++)
        {
            var layer = new Neuron[hiddenLayerSize];

            for (var j = 0; j < hiddenLayerSize; j++)
            {
                var inputs = i == 0 ? inputsCount : hiddenLayerSize;
                layer[j] = NeuronFactory.CreateNeuron(inputs);
            }
            
            hiddenLayers[i] = layer;
        }
        
        return hiddenLayers;
    }
}