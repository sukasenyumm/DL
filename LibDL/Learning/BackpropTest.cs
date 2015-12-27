using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using LibDL.Interface;
using LibDL;
using LibDL.ActivationFunction;

namespace LibDL.Learning
{

    public class BackpropTest
    {
        // network to teach
        private NeuralNetwork network;
        // learning rate
        private double learningRate = 0.1;
        // momentum
        private double momentum = 0.0;

        // neuron's errors
        private double[][] neuronErrors = null;
        // weight's updates
        private double[][][] weightsUpdates = null;
        // threshold's updates
        private double[][] thresholdsUpdates = null;

        /// <summary>
        /// Learning rate, [0, 1].
        /// </summary>
        /// 
        /// <remarks><para>The value determines speed of learning.</para>
        /// 
        /// <para>Default value equals to <b>0.1</b>.</para>
        /// </remarks>
        ///
        public double LearningRate
        {
            get { return learningRate; }
            set
            {
                learningRate = Math.Max(0.0, Math.Min(1.0, value));
            }
        }

        /// <summary>
        /// Momentum, [0, 1].
        /// </summary>
        /// 
        /// <remarks><para>The value determines the portion of previous weight's update
        /// to use on current iteration. Weight's update values are calculated on
        /// each iteration depending on neuron's error. The momentum specifies the amount
        /// of update to use from previous iteration and the amount of update
        /// to use from current iteration. If the value is equal to 0.1, for example,
        /// then 0.1 portion of previous update and 0.9 portion of current update are used
        /// to update weight's value.</para>
        /// 
        /// <para>Default value equals to <b>0.0</b>.</para>
        ///	</remarks>
        /// 
        public double Momentum
        {
            get { return momentum; }
            set
            {
                momentum = Math.Max(0.0, Math.Min(1.0, value));
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="BackPropagationLearning"/> class.
        /// </summary>
        /// 
        /// <param name="network">Network to teach.</param>
        /// 
        public BackpropTest(NeuralNetwork network)
        {
            this.network = network;

            // create error and deltas arrays
            neuronErrors = new double[network.UsualLayers.Length][];
            weightsUpdates = new double[network.UsualLayers.Length][][];
            thresholdsUpdates = new double[network.UsualLayers.Length][];

            // initialize errors and deltas arrays for each layer
            for (int i = 0; i < network.UsualLayers.Length; i++)
            {
                UsualNetworkLayer layer = network.UsualLayers[i];

                neuronErrors[i] = new double[layer.Neurons.Length];
                weightsUpdates[i] = new double[layer.Neurons.Length][];
                thresholdsUpdates[i] = new double[layer.Neurons.Length];

                // for each neuron
                for (int j = 0; j < weightsUpdates[i].Length; j++)
                {
                    weightsUpdates[i][j] = new double[layer.InputsCount];
                }
            }
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
            // compute the network's output
            network.Compute(input);

            // calculate network error
            double error = CalculateError(output);

            // calculate weights updates
            CalculateUpdates(input);

            // update the network
            UpdateNetwork();

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
            double error = 0.0;

            // run learning procedure for all samples
            for (int i = 0; i < input.Length; i++)
            {
                error += Run(input[i], output[i]);
            }

            // return summary error
            return error;
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
            // current and the next layers
            UsualNetworkLayer layer, layerNext;
            // current and the next errors arrays
            double[] errors, errorsNext;
            // error values
            double error = 0, e, sum;
            // neuron's output value
            double output;
            // layers count
            int layersCount = network.UsualLayers.Length;

            // assume, that all neurons of the network have the same activation function
            IActivationFunction function = (network.UsualLayers[0].Neurons[0] as UsualNeuron).ActivationFunction;
            //IActivationFunction function = new SigmoidFunction(0.1);

            // calculate error values for the last layer first
            layer = network.UsualLayers[layersCount - 1];
            errors = neuronErrors[layersCount - 1];

            for (int i = 0; i < layer.Neurons.Length; i++)
            {
                output = layer.Neurons[i].Output;
                // error of the neuron
                e = desiredOutput[i] - output;
                // error multiplied with activation function's derivative
                errors[i] = e * function.DerivativeOutput(output);
                // squre the error and sum it
                error += (e * e);
            }

            // calculate error values for other layers
            for (int j = layersCount - 2; j >= 0; j--)
            {
                layer = network.UsualLayers[j];
                layerNext = network.UsualLayers[j + 1];
                errors = neuronErrors[j];
                errorsNext = neuronErrors[j + 1];

                // for all neurons of the layer
                for (int i = 0; i < layer.Neurons.Length; i++)
                {
                    sum = 0.0;
                    // for all neurons of the next layer
                    for (int k = 0; k < layerNext.Neurons.Length; k++)
                    {
                        sum += errorsNext[k] * layerNext.Neurons[k].Weights[i];
                    }
                    errors[i] = sum * function.DerivativeOutput(layer.Neurons[i].Output);
                }
            }

            // return squared error of the last layer divided by 2
            return error / 2.0;
        }

        /// <summary>
        /// Calculate weights updates.
        /// </summary>
        /// 
        /// <param name="input">Network's input vector.</param>
        /// 
        private void CalculateUpdates(double[] input)
        {
            // 1 - calculate updates for the first layer
            UsualNetworkLayer layer = network.UsualLayers[0];
            double[] errors = neuronErrors[0];
            double[][] layerWeightsUpdates = weightsUpdates[0];
            double[] layerThresholdUpdates = thresholdsUpdates[0];

            // cache for frequently used values
            double cachedMomentum = learningRate * momentum;
            double cached1mMomentum = learningRate * (1 - momentum);
            double cachedError;

            // for each neuron of the layer
            for (int i = 0; i < layer.Neurons.Length; i++)
            {
                cachedError = errors[i] * cached1mMomentum;
                double[] neuronWeightUpdates = layerWeightsUpdates[i];

                // for each weight of the neuron
                for (int j = 0; j < neuronWeightUpdates.Length; j++)
                {
                    // calculate weight update
                    neuronWeightUpdates[j] = cachedMomentum * neuronWeightUpdates[j] + cachedError * input[j];
                }

                // calculate threshold update
                layerThresholdUpdates[i] = cachedMomentum * layerThresholdUpdates[i] + cachedError;
            }

            // 2 - for all other layers
            for (int k = 1; k < network.UsualLayers.Length; k++)
            {
                UsualNetworkLayer layerPrev = network.UsualLayers[k - 1];
                layer = network.UsualLayers[k];
                errors = neuronErrors[k];
                layerWeightsUpdates = weightsUpdates[k];
                layerThresholdUpdates = thresholdsUpdates[k];

                // for each neuron of the layer
                for (int i = 0; i < layer.Neurons.Length; i++)
                {
                    cachedError = errors[i] * cached1mMomentum;
                    double[] neuronWeightUpdates = layerWeightsUpdates[i];

                    // for each synapse of the neuron
                    for (int j = 0; j < neuronWeightUpdates.Length; j++)
                    {
                        // calculate weight update
                        neuronWeightUpdates[j] = cachedMomentum * neuronWeightUpdates[j] + cachedError * layerPrev.Neurons[j].Output;
                    }

                    // calculate threshold update
                    layerThresholdUpdates[i] = cachedMomentum * layerThresholdUpdates[i] + cachedError;
                }
            }
        }

        /// <summary>
        /// Update network's weights.
        /// </summary>
        /// 
        private void UpdateNetwork()
        {
            // current neuron
            UsualNeuron neuron;
            // current layer
            UsualNetworkLayer layer;
            // layer's weights updates
            double[][] layerWeightsUpdates;
            // layer's thresholds updates
            double[] layerThresholdUpdates;
            // neuron's weights updates
            double[] neuronWeightUpdates;

            // for each layer of the network
            for (int i = 0; i < network.UsualLayers.Length; i++)
            {
                layer = network.UsualLayers[i];
                layerWeightsUpdates = weightsUpdates[i];
                layerThresholdUpdates = thresholdsUpdates[i];

                // for each neuron of the layer
                for (int j = 0; j < layer.Neurons.Length; j++)
                {
                    neuron = layer.Neurons[j] as UsualNeuron;
                    neuronWeightUpdates = layerWeightsUpdates[j];

                    // for each weight of the neuron
                    for (int k = 0; k < neuron.Weights.Length; k++)
                    {
                        // update weight
                        neuron.Weights[k] += neuronWeightUpdates[k];
                    }
                    // update treshold
                    neuron.Threshold += layerThresholdUpdates[j];
                }
            }
        }
    }
}

