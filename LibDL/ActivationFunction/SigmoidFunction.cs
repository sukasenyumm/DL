
namespace LibDL.ActivationFunction
{
    using System;
    using LibDL.Interface;

    /// <summary>
    /// Sigmoid activation function.
    /// </summary>
    ///
    /// <remarks><para>The class represents sigmoid activation function with
    /// the next expression:
    /// <code lang="none">
    ///                1
    /// f(x) = ------------------
    ///        1 + exp(-alpha * x)
    ///
    ///           alpha * exp(-alpha * x )
    /// f'(x) = ---------------------------- = alpha * f(x) * (1 - f(x))
    ///           (1 + exp(-alpha * x))^2
    /// </code>
    /// </para>
    ///
    /// <para>Output range of the function: <b>[0, 1]</b>.</para>
    /// 
    /// <para>Functions graph:</para>
    /// <img src="img/neuro/sigmoid.bmp" width="242" height="172" />
    /// </remarks>
    /// 
    [Serializable]
    public class SigmoidFunction : IActivationFunction, ICloneable
    {
        // sigmoid's alpha value
        private double alpha = 2;

        /// <summary>
        /// Sigmoid's alpha value.
        /// </summary>
        /// 
        /// <remarks><para>The value determines steepness of the function. Increasing value of
        /// this property changes sigmoid to look more like a threshold function. Decreasing
        /// value of this property makes sigmoid to be very smooth (slowly growing from its
        /// minimum value to its maximum value).</para>
        ///
        /// <para>Default value is set to <b>2</b>.</para>
        /// </remarks>
        /// 
        public double Alpha
        {
            get { return alpha; }
            set { alpha = value; }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="SigmoidFunction"/> class.
        /// </summary>
        public SigmoidFunction() { }

        /// <summary>
        /// Initializes a new instance of the <see cref="SigmoidFunction"/> class.
        /// </summary>
        /// 
        /// <param name="alpha">Sigmoid's alpha value.</param>
        /// 
        public SigmoidFunction(double alpha)
        {
            this.alpha = alpha;
        }

        public object Clone()
        {
            return new SigmoidFunction(alpha);
        }

        public double ActivationFunction(double x)
        {
            return (1 / (1 + Math.Exp(-alpha * x)));
        }

        public double DerivativeInput(double x)
        {
            double y = ActivationFunction(x);

            return (alpha * y * (1 - y));
        }

        public double DerivativeOutput(double y)
        {
            return (alpha * y * (1 - y));
        }

        public double GenerateFromInput(double x)
        {
            throw new NotImplementedException();
        }

        public double GenerateFromOutput(double y)
        {
            throw new NotImplementedException();
        }
    }
}
