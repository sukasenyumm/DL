namespace LibDL.Interface
{
    public interface IActivationFunction
    {
        // non-Stochastic
        double ActivationFunction(double x);
        double DerivativeInput(double x);
        double DerivativeOutput(double y);
        //Stochastic
        double GenerateFromInput(double x);
        double GenerateFromOutput(double y);
    }
}
