namespace NeuralNetwork;

public class Neuron(double[] weights, double bias)
{
    private double? _scaledError;
    private double[] _weights = weights;
    private double _bias = bias;
    private double? _output;

    public double ActivateAndGetOutput(double[] inputs)
    {
        Activate(inputs);
        
        return _output!.Value;
    }

    public void SetScaledError(double error)
    {
        if (_output == null)
        {
            throw new InvalidOperationException("Neuron has not been activated");
        }
        
        _scaledError = error * Derivative((double)_output);
    }
    
    private void Activate(double[] inputs)
    {
        if (inputs.Length != _weights.Length)
        {
            throw new ArgumentException("Neuron must have the same number of inputs and weight");
        }

        var weightedInputs = inputs.Select((input, index) => input * _weights[index]).Sum() + _bias;

        _output = Sigmoid(weightedInputs);
    }
    
    private static double Sigmoid(double x)
    {
        return 1.0 / (1 + Math.Exp(-x));
    }
    
    private static double Derivative(double y)
    {
        return y * (1 - y);
    }
}