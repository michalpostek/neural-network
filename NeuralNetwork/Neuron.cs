namespace NeuralNetwork;

public class Neuron(double[] weights, double bias)
{
    public double[] Weights { get; } = weights;
    public double Bias { set; get; } = bias;
    public double? Output { get; private set; }
    public double? Delta;

    public double ActivateAndGetOutput(double[] inputs)
    {
        if (inputs.Length != Weights.Length)
        {
            throw new ArgumentException("Neuron must have the same number of inputs and weight");
        }

        var weightedInputs = inputs.Select((input, index) => input * Weights[index]).Sum() + Bias;
        Output = Sigmoid(weightedInputs);
        
        return Output!.Value;
    }

    public void SetDelta(double error)
    {
        Delta = error * Derivative(Output!.Value);
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