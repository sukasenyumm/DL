
using System;
using LibDL.Interface;
using LibDL;
using System.Threading;
using System.Threading.Tasks;

namespace LibDL.Learning
{

    /// <summary>
    /// Resilient Backpropagation learning algorithm.
    /// </summary>
    /// 
    /// <remarks><para>This class implements the resilient backpropagation (RProp)
    /// learning algorithm. The RProp learning algorithm is one of the fastest learning
    /// algorithms for feed-forward learning networks which use only first-order
    /// information.</para>
    /// 
    /// <para>Sample usage (training network to calculate XOR function):</para>
    /// <code>
    /// // initialize input and output values
    /// double[][] input = new double[4][] {
    ///     new double[] {0, 0}, new double[] {0, 1},
    ///     new double[] {1, 0}, new double[] {1, 1}
    /// };
    /// double[][] output = new double[4][] {
    ///     new double[] {0}, new double[] {1},
    ///     new double[] {1}, new double[] {0}
    /// };
    /// // create neural network
    /// ActivationNetwork   network = new ActivationNetwork(
    ///     SigmoidFunction( 2 ),
    ///     2, // two inputs in the network
    ///     2, // two neurons in the first layer
    ///     1 ); // one neuron in the second layer
    /// // create teacher
    /// ResilientBackpropagationLearning teacher = new ResilientBackpropagationLearning( network );
    /// // loop
    /// while ( !needToStop )
    /// {
    ///     // run epoch of learning procedure
    ///     double error = teacher.RunEpoch( input, output );
    ///     // check error value to see if we need to stop
    ///     // ...
    /// }
    /// </code>
    /// </remarks>
    /// 
    public class ResilientBackPropagationFineTuning : IDisposable
    {
        private NeuralNetwork network;

        private double delta0 = 0.0125;
        private double deltaMax = 50.0;
        private double deltaMin = 1e-6;

        private const double etaMinus =0.5;
        private double etaPlus = 1.2;

        //private double[][] neuronErrors = null;

        // update values, also known as deltas
        private double[][][] weightsUpdates = null;
        private double[][] thresholdsUpdates = null;

        // current and previous gradient values
        private double[][][] weightsDerivatives = null;
        private double[][] thresholdsDerivatives = null;

        private double[][][] weightsPreviousDerivatives = null;
        private double[][] thresholdsPreviousDerivatives = null;

        private Object lockNetwork = new Object();
        private ThreadLocal<double[][]> networkErrors;
        private ThreadLocal<double[][]> networkOutputs;
        /// <summary>
        /// Learning rate.
        /// </summary>
        /// 
        /// <remarks><para>The value determines speed of learning.</para>
        /// 
        /// <para>Default value equals to <b>0.0125</b>.</para>
        /// </remarks>
        ///
        //public double LearningRate
        //{
        //    get { return learningRate; }
        //    set
        //    {
        //        learningRate = value;
        //        ResetUpdates(learningRate);
        //    }
        //}

        /// <summary>
        /// Initializes a new instance of the <see cref="ResilientBackpropagationLearning"/> class.
        /// </summary>
        /// 
        /// <param name="network">Network to teach.</param>
        /// 
        public ResilientBackPropagationFineTuning(NeuralNetwork network)
        {
            this.network = network;

            int layersCount = network.Layers.Length;

            networkOutputs = new ThreadLocal<double[][]>(() => new double[layersCount][]);

            networkErrors = new ThreadLocal<double[][]>(() =>
            {
                var e = new double[network.Layers.Length][];
                for (int i = 0; i < e.Length; i++)
                    e[i] = new double[network.Layers[i].Neurons.Length];
                return e;
            });

            weightsDerivatives = new double[layersCount][][];
            thresholdsDerivatives = new double[layersCount][];

            weightsPreviousDerivatives = new double[layersCount][][];
            thresholdsPreviousDerivatives = new double[layersCount][];

            weightsUpdates = new double[layersCount][][];
            thresholdsUpdates = new double[layersCount][];

            // initialize errors, derivatives and steps
            for (int i = 0; i < network.Layers.Length; i++)
            {
                NetworkLayer layer = network.Layers[i];
                int neuronsCount = layer.Neurons.Length;

                weightsDerivatives[i] = new double[neuronsCount][];
                weightsPreviousDerivatives[i] = new double[neuronsCount][];
                weightsUpdates[i] = new double[neuronsCount][];

                thresholdsDerivatives[i] = new double[neuronsCount];
                thresholdsPreviousDerivatives[i] = new double[neuronsCount];
                thresholdsUpdates[i] = new double[neuronsCount];

                // for each neuron
                for (int j = 0; j < layer.Neurons.Length; j++)
                {
                    weightsDerivatives[i][j] = new double[layer.InputsCount];
                    weightsPreviousDerivatives[i][j] = new double[layer.InputsCount];
                    weightsUpdates[i][j] = new double[layer.InputsCount];
                }
            }

            // intialize steps
            ResetUpdates(delta0);
        }

        /// <summary>
        /// Runs learning iteration.
        /// </summary>
        /// 
        /// <param name="input">Input vector.</param>
        /// <param name="output">Desired output vector.</param>
        /// 
        /// <returns>Returns squared error (difference between current network's output and
        /// desired output) divided by 2.</returns>
        /// 
        /// <remarks><para>Runs one learning iteration and updates neuron's
        /// weights.</para></remarks>
        ///
        public double Run(double[] input, double[] output)
        {
            // zero gradient
            ResetGradient();

            // compute the network's output
            network.Compute(input);

            // calculate network error
            double error = CalculateError(output);

            // calculate weights updates
            CalculateGradient(input);

            // update the network
            UpdateNetwork();

            // return summary error
            return error;
        }

        /// <summary>
        /// Runs learning epoch.
        /// </summary>
        /// 
        /// <param name="input">Array of input vectors.</param>
        /// <param name="output">Array of output vectors.</param>
        /// 
        /// <returns>Returns summary learning error for the epoch. See <see cref="Run"/>
        /// method for details about learning error calculation.</returns>
        /// 
        /// <remarks><para>The method runs one learning epoch, by calling <see cref="Run"/> method
        /// for each vector provided in the <paramref name="input"/> array.</para></remarks>
        /// 
        public double RunEpoch(double[][] input, double[][] output)
        {
            // Zero gradient
            ResetGradient();

            Object lockSum = new Object();
            double sumOfSquaredErrors = 0;


            // For all examples in batch
            Parallel.For(0, input.Length,
#if DEBUG
 new ParallelOptions() { MaxDegreeOfParallelism = Environment.ProcessorCount / 2 + Environment.ProcessorCount / 3 },
#endif

                // Initialize
                () => 0.0,

                // Map
                (i, loopState, partialSum) =>
                {

                    lock (lockNetwork)
                    {
                        // Compute a forward pass
                        network.Compute(input[i]);

                        // Copy network outputs to local thread
                        var networkOutputs = this.networkOutputs.Value;
                        for (int j = 0; j < networkOutputs.Length; j++)
                            networkOutputs[j] = network.Layers[j].Output;
                    }

                    // Calculate and accumulate network error
                    partialSum += CalculateError(output[i]);

                    // Calculate weights updates
                    CalculateGradient(input[i]);

                    return partialSum;
                },

                // Reduce
                (partialSum) =>
                {
                    lock (lockSum) sumOfSquaredErrors += partialSum;
                }
            );


            // Update the network
            UpdateNetwork();


            return sumOfSquaredErrors;
        }

        /// <summary>
        /// Resets current weight and threshold derivatives.
        /// </summary>
        /// 
        private void ResetGradient()
        {
            Parallel.For(0, weightsDerivatives.Length, i =>
            {
                for (int j = 0; j < weightsDerivatives[i].Length; j++)
                    Array.Clear(weightsDerivatives[i][j], 0, weightsDerivatives[i][j].Length);
                Array.Clear(thresholdsDerivatives[i], 0, thresholdsDerivatives[i].Length);
            });
        }

        /// <summary>
        /// Resets the current update steps using the given learning rate.
        /// </summary>
        /// 
        private void ResetUpdates(double rate)
        {
            Parallel.For(0, weightsUpdates.Length, i =>
            {
                for (int j = 0; j < weightsUpdates[i].Length; j++)
                    for (int k = 0; k < weightsUpdates[i][j].Length; k++)
                        weightsUpdates[i][j][k] = rate;

                for (int j = 0; j < thresholdsUpdates[i].Length; j++)
                    thresholdsUpdates[i][j] = rate;
            });
        }

        /// <summary>
        /// Update network's weights.
        /// </summary>
        /// 
        private void UpdateNetwork()
        {
            double[][] layerWeightsUpdates;

            double[] layerThresholdUpdates;
            double[] neuronWeightUpdates;

            double[][] layerWeightsDerivatives;
            double[] layerThresholdDerivatives;
            double[] neuronWeightDerivatives;

            double[][] layerPreviousWeightsDerivatives;
            double[] layerPreviousThresholdDerivatives;
            double[] neuronPreviousWeightDerivatives;

            double delta;
            double deltaT;

            // for each layer of the network
            for (int i = 0; i < network.Layers.Length; i++)
            {
                NetworkLayer layer = network.Layers[i] as NetworkLayer;

                layerWeightsUpdates = weightsUpdates[i];
                layerThresholdUpdates = thresholdsUpdates[i];

                layerWeightsDerivatives = weightsDerivatives[i];
                layerThresholdDerivatives = thresholdsDerivatives[i];

                layerPreviousWeightsDerivatives = weightsPreviousDerivatives[i];
                layerPreviousThresholdDerivatives = thresholdsPreviousDerivatives[i];

                // for each neuron of the layer
                for (int j = 0; j < layer.Neurons.Length; j++)
                {
                    Neuron neuron = layer.Neurons[j] as Neuron;
                    neuronWeightUpdates = layerWeightsUpdates[j];
                    neuronWeightDerivatives = layerWeightsDerivatives[j];
                    neuronPreviousWeightDerivatives = layerPreviousWeightsDerivatives[j];

                    double S = 0;

                    // for each weight of the neuron
                    for (int k = 0; k < neuron.InputsCount; k++)
                    {
                        S = neuronPreviousWeightDerivatives[k] * neuronWeightDerivatives[k];

                        if (S > 0)
                        {
                            delta = neuronWeightUpdates[k] * etaPlus;
                            neuronWeightUpdates[k] = Math.Min(delta, deltaMax);
                            neuron.Weights[k] -= Math.Sign(neuronWeightDerivatives[k]) * neuronWeightUpdates[k];
                            //neuronPreviousWeightDerivatives[k] = neuronWeightDerivatives[k];
                        }
                        else if (S < 0)
                        {
                            delta = neuronWeightUpdates[k] * etaMinus;
                            neuronWeightUpdates[k] = Math.Max(delta, deltaMin);
                            neuronWeightDerivatives[k] = 0;
                        }
                        else
                        {
                            delta = neuronWeightUpdates[k];
                            neuron.Weights[k] -= Math.Sign(neuronWeightDerivatives[k]) * neuronWeightUpdates[k];
                            //neuronPreviousWeightDerivatives[k] = neuronWeightDerivatives[k];
                        }

                        neuronWeightUpdates[k] = delta;
                        neuronPreviousWeightDerivatives[k] = neuronWeightDerivatives[k];

                    }

                    // update treshold
                    S = layerPreviousThresholdDerivatives[j] * layerThresholdDerivatives[j];

                    if (S > 0)
                    {
                        deltaT = layerThresholdUpdates[j] * etaPlus;
                        layerThresholdUpdates[j] = Math.Min(deltaT, deltaMax);
                        neuron.Threshold -= Math.Sign(layerThresholdDerivatives[j]) * layerThresholdUpdates[j];
                        //layerPreviousThresholdDerivatives[j] = layerThresholdDerivatives[j];
                    }
                    else if (S < 0)
                    {
                        deltaT = layerThresholdUpdates[j] * etaMinus;
                        layerThresholdUpdates[j] = Math.Max(deltaT, deltaMin);
                        layerThresholdDerivatives[j] = 0;
                    }
                    else
                    {
                        deltaT = layerThresholdUpdates[j];
                        neuron.Threshold -= Math.Sign(layerThresholdDerivatives[j]) * layerThresholdUpdates[j];
                        //ayerPreviousThresholdDerivatives[j] = layerThresholdDerivatives[j];
                    }

                    layerThresholdUpdates[j] = deltaT;
                    layerPreviousThresholdDerivatives[j] = layerThresholdDerivatives[j];
                }
            }
        }

        /// <summary>
        /// Calculates error values for all neurons of the network.
        /// </summary>
        /// 
        /// <param name="desiredOutput">Desired output vector.</param>
        /// 
        /// <returns>Returns summary squared error of the last layer divided by 2.</returns>
        /// 
        private double CalculateError(double[] desiredOutput)
        {
            double sumOfSquaredErrors = 0.0;
            int layersCount = network.Layers.Length;

            double[][] networkErrors = this.networkErrors.Value;
            double[][] networkOutputs = this.networkOutputs.Value;

            // Assume that all network neurons have the same activation function
            var function = (this.network.Layers[0].Neurons[0] as Neuron)
                .ActivationFunction;

            // 1. Calculate error values for last layer first.
            double[] layerOutputs = networkOutputs[layersCount - 1];
            double[] errors = networkErrors[layersCount - 1];

            for (int i = 0; i < errors.Length; i++)
            {
                double output = layerOutputs[i];
                double e = output - desiredOutput[i];
                errors[i] = e * function.DerivativeOutput(output);
                sumOfSquaredErrors += e * e;
            }

            // 2. Calculate errors for all other layers
            for (int j = layersCount - 2; j >= 0; j--)
            {
                errors = networkErrors[j];
                layerOutputs = networkOutputs[j];

                NetworkLayer nextLayer = network.Layers[j + 1] as NetworkLayer;
                double[] nextErrors = networkErrors[j + 1];

                // For all neurons of this layer
                for (int i = 0; i < errors.Length; i++)
                {
                    double sum = 0.0;

                    // For all neurons of the next layer
                    for (int k = 0; k < nextErrors.Length; k++)
                        sum += nextErrors[k] * nextLayer.Neurons[k].Weights[i];

                    errors[i] = sum * function.DerivativeOutput(layerOutputs[i]);
                }
            }

            return sumOfSquaredErrors / 2.0;
        }

        /// <summary>
        /// Calculate weights updates
        /// </summary>
        /// 
        /// <param name="input">Network's input vector.</param>
        /// 
        private void CalculateGradient(double[] input)
        {
            double[][] networkErrors = this.networkErrors.Value;
            double[][] networkOutputs = this.networkOutputs.Value;
            // 1. calculate updates for the first layer
            //NetworkLayer layer = network.Layers[0] as NetworkLayer;
            double[] weightErrors = networkErrors[0];
            double[][] layerWeightsDerivatives = weightsDerivatives[0];
            double[] layerThresholdDerivatives = thresholdsDerivatives[0];

            // For each neuron of the last layer
            for (int i = 0; i < weightErrors.Length; i++)
            {
                double[] neuronWeightDerivatives = layerWeightsDerivatives[i];

                lock (neuronWeightDerivatives)
                {
                    // For each weight in the neuron
                    for (int j = 0; j < input.Length; j++)
                        neuronWeightDerivatives[j] += weightErrors[i] * input[j];
                    layerThresholdDerivatives[i] += weightErrors[i];
                }
            }

            // 2. Calculate for all other layers in a chain
            for (int k = 1; k < weightsDerivatives.Length; k++)
            {
                weightErrors = networkErrors[k];

                layerWeightsDerivatives = weightsDerivatives[k];
                layerThresholdDerivatives = thresholdsDerivatives[k];

                double[] layerPrev = networkOutputs[k - 1];

                // For each neuron in the current layer
                for (int i = 0; i < layerWeightsDerivatives.Length; i++)
                {
                    double[] neuronWeightDerivatives = layerWeightsDerivatives[i];

                    lock (neuronWeightDerivatives)
                    {
                        // For each weight of the neuron
                        for (int j = 0; j < neuronWeightDerivatives.Length; j++)
                            neuronWeightDerivatives[j] += weightErrors[i] * layerPrev[j];
                        layerThresholdDerivatives[i] += weightErrors[i];
                    }
                }
            }
        }

        #region IDisposable Members

        /// <summary>
        ///   Performs application-defined tasks associated with freeing,
        ///   releasing, or resetting unmanaged resources.
        /// </summary>
        /// 
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        ///   Releases unmanaged resources and performs other cleanup operations before
        ///   the <see cref="ParallelResilientBackpropagationLearning"/> is reclaimed by garbage
        ///   collection.
        /// </summary>
        /// 
        ~ResilientBackPropagationFineTuning()
        {
            Dispose(false);
        }

        /// <summary>
        ///   Releases unmanaged and - optionally - managed resources
        /// </summary>
        /// 
        /// <param name="disposing"><c>true</c> to release both managed 
        /// and unmanaged resources; <c>false</c> to release only unmanaged
        /// resources.</param>
        /// 
        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                // free managed resources
                if (networkErrors != null)
                {
                    networkErrors.Dispose();
                    networkErrors = null;
                }
                if (networkOutputs != null)
                {
                    networkOutputs.Dispose();
                    networkOutputs = null;
                }
            }
        }

        #endregion
    }
}
