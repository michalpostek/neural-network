namespace NeuralNetwork;

public static class NeuronFactory
{
    private static readonly Random Random = new();
    
    public static Neuron[] InitOutputLayer(int hiddenLayerSize, int outputsCount)
    {
        var layer = new Neuron[outputsCount];

        for (var i = 0; i < outputsCount; i++)
        {
            layer[i] = CreateNeuron(hiddenLayerSize);
        }
        
        return layer;
    }

    public static Neuron[][] InitHiddenLayers(int inputsCount, int hiddenLayersCount, int hiddenLayerSize)
    {
        var hiddenLayers = new Neuron[hiddenLayersCount][];

        for (var i = 0; i < hiddenLayersCount; i++)
        {
            var layer = new Neuron[hiddenLayerSize];

            for (var j = 0; j < hiddenLayerSize; j++)
            {
                var inputs = i == 0 ? inputsCount : hiddenLayerSize;
                layer[j] = CreateNeuron(inputs);
            }
            
            hiddenLayers[i] = layer;
        }
        
        return hiddenLayers;
    }
    
    private static Neuron CreateNeuron(int weights)
    {
        return new Neuron(RandomWeights(weights), RandomWeight());
    }
    
    private static double[] RandomWeights(int length)
    {
        var weights = new double[length];

        for (var i = 0; i < length; i++)
        {
            weights[i] = RandomWeight();
        }

        return weights;
    }

    private static double RandomWeight()
    {
        return Random.NextDouble() * 2 - 1;
    } 
}