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
    public class Backpropagation: IDisposable
    {
        //buat jaringan syaraf dbn untuk nantinya diberikan dari proses layer-wise unsuprvise learning
        private NeuralNetwork network;
        // learning rate
        private double learningRate = 0.1;
        public double LearningRate
        {
            get { return learningRate; }
            set
            {
                learningRate = Math.Max(0.0, Math.Min(1.0, value));
            }
        }
        // momentum
        private double momentum = 0.0;
        public double Momentum
        {
            get { return momentum; }
            set
            {
                momentum = Math.Max(0.0, Math.Min(1.0, value));
            }
        }
        // definisikan error untuk setiap neuron/node
        private Object lockNetwork = new Object();
        // definisikan error untuk setiap neuron/node
        private ThreadLocal<double[][]> networkOutputs;
        private ThreadLocal<double[][]> neuronErrors;
        // definisikan update bobot
        private double[][][] weightsUpdates = null;
        // definisikan update thresholds/bias
        private double[][] thresholdsUpdates = null;

        public Backpropagation(NeuralNetwork network)
        {
            this.network = network;

            networkOutputs = new ThreadLocal<double[][]>(() => new double[network.Layers.Length][]);

            neuronErrors = new ThreadLocal<double[][]>(() =>
            {
                var e = new double[network.Layers.Length][];
                for (int i = 0; i < e.Length; i++)
                    e[i] = new double[network.Layers[i].Neurons.Length];
                return e;
            });

            //buar array untuk error dan delta
            weightsUpdates = new double[network.Layers.Length][][];
            thresholdsUpdates = new double[network.Layers.Length][];

            //inisilaisasi array error dan delta pada setiap layer
            for (int i = 0; i < network.Layers.Length; i++)
            {
                NetworkLayer layer = network.Layers[i];

                weightsUpdates[i] = new double[layer.Neurons.Length][];
                thresholdsUpdates[i] = new double[layer.Neurons.Length];

                // untuk setiap neuron
                for (int j = 0; j < weightsUpdates[i].Length; j++)
                {
                    weightsUpdates[i][j] = new double[layer.InputsCount];
                }
            }
        }

        private void ResetGradient()
        {
            Parallel.For(0, weightsUpdates.Length, i =>
            {
                for (int j = 0; j < weightsUpdates[i].Length; j++)
                    Array.Clear(weightsUpdates[i][j], 0, weightsUpdates[i][j].Length);
                Array.Clear(thresholdsUpdates[i], 0, thresholdsUpdates[i].Length);
            });
        }


        public double[][][] GetLayerIO(double[][][] batches)
        {
            if (network.Layers.Length == 0)
                return batches;

            var outputBatches = new double[batches.Length][][];

            for (int j = 0; j < batches.Length; j++)
            {
                int batchSize = batches[j].Length;

                double[][] inputs = batches[j];
                double[][] outputs = new double[batchSize][];

                outputBatches[j] = inputs;
            }

            return outputBatches;
        }

        public double RunEpoch(double[][][] batchesIn, double[][][] batchesOut)
        {
           
            // Learn the layer using data
            double error = 0;
            for (int i = 0; i < batchesIn.Length; i++)
                error += this.RunEpoch(batchesIn[i], batchesOut[i]);

            return error;
        }
       
        public double RunEpoch(double[][] input, double[][] output)
        {
            //ResetGradient();
            double error = 0.0;
            Object lockSum = new Object();

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
                    CalculateUpdates(input[i]);

                    // Update the network
                    UpdateNetwork();

                    return partialSum;
                },

                // Reduce
                (partialSum) =>
                {
                    lock (lockSum) error += partialSum;
                }
            );

            return error;

            // run learning procedure for all samples
            //for (int i = 0; i < input.Length; i++)
            //{
            //    error += Run(input[i], output[i]);
            //}

            // return summary error
        }
        private double CalculateError(double[] desiredOutput)
        {
            //karena backprop berjalan mundur definisikan layer saat ini dan selanjutnya beserta error yang dihasilkan juga
            double[] layer;
            double[] errors, errorsNext;

            //definisikan error untuk setiap neuron
            double error = 0, sum;
            //hitung banyaknya layer pada jaringan
            int layersCount = network.Layers.Length;

            //asumsikan bahwa semua neuron pada layer jaringan memiliki fungsi aktivasi yang sama
            IActivationFunction function = (this.network.Layers[0].Neurons[0] as Neuron).ActivationFunction;
            //SigmoidFunction function = new SigmoidFunction(alpha: 0.0001); //derivative2 = derivativeoutput

            double[][] neuErrors = this.neuronErrors.Value;
            double[][] netOutputs = this.networkOutputs.Value;

            //pertama, hitung error pada output layer
            layer = netOutputs[layersCount - 1];
            errors = neuErrors[layersCount - 1];

            for (int i = 0; i < errors.Length; i++)
            {
                //definisikan nilai ouput pada neuron
                double output = layer[i];
                // error of the neuron
                double e = desiredOutput[i] - output;
                // error multiplied with activation function's derivative
                errors[i] = e * function.DerivativeOutput(output);
                // squre the error and sum it
                error += (e * e);
            }

            // calculate error values for other layers
            for (int j = layersCount - 2; j >= 0; j--)
            {
                layer = netOutputs[j];
                errors = neuErrors[j];
                errorsNext = neuErrors[j + 1];
                NetworkLayer nextLayer = network.Layers[j + 1] as NetworkLayer;
                

                // for all neurons of the layer
                for (int i = 0; i < errors.Length; i++)
                {
                    sum = 0.0;
                    // for all neurons of the next layer
                    for (int k = 0; k < errorsNext.Length; k++)
                    {
                        sum += errorsNext[k] * nextLayer.Neurons[k].Weights[i];
                    }
                    errors[i] = sum * function.DerivativeOutput(layer[i]);
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
           
            double[][] neuErrors = this.neuronErrors.Value;
            double[][] netOutputs = this.networkOutputs.Value;
            double[] layer = netOutputs[0];
            double[] errors = neuErrors[0];           

            double[][] layerWeightsUpdates = weightsUpdates[0];
            double[] layerThresholdUpdates = thresholdsUpdates[0];

            // cache for frequently used values
            double cachedMomentum = learningRate * momentum;
            double cached1mMomentum = learningRate * (1 - momentum);
            double cachedError;

            // for each neuron of the layer
            for (int i = 0; i < errors.Length; i++)
            {
                cachedError = errors[i] * cached1mMomentum;
                double[] neuronWeightUpdates = layerWeightsUpdates[i];

                lock (neuronWeightUpdates)
                {
                    // for each weight of the neuron
                    for (int j = 0; j < neuronWeightUpdates.Length; j++)
                    {
                        // calculate weight update
                        neuronWeightUpdates[j] = cachedMomentum * neuronWeightUpdates[j] + cachedError * input[j];
                    }

                    // calculate threshold update
                    layerThresholdUpdates[i] = cachedMomentum * layerThresholdUpdates[i] + cachedError;
                }
            }

            // 2 - for all other layers
            for (int k = 1; k < weightsUpdates.Length; k++)
            {
                double[] layerPrev = netOutputs[k - 1];
                layer = netOutputs[k];
                errors = neuErrors[k];
                layerWeightsUpdates = weightsUpdates[k];
                layerThresholdUpdates = thresholdsUpdates[k];

                // for each neuron of the layer
                for (int i = 0; i < layerWeightsUpdates.Length; i++)
                {
                    cachedError = errors[i] * cached1mMomentum;
                    double[] neuronWeightUpdates = layerWeightsUpdates[i];

                    lock (neuronWeightUpdates)
                    {
                        // for each synapse of the neuron
                        for (int j = 0; j < neuronWeightUpdates.Length; j++)
                        {
                            // calculate weight update
                            neuronWeightUpdates[j] = cachedMomentum * neuronWeightUpdates[j] + cachedError * layerPrev[j];
                        }
                        // calculate threshold update
                        layerThresholdUpdates[i] = cachedMomentum * layerThresholdUpdates[i] + cachedError;
                    }
                    
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
            Neuron neuron;
            // current layer
            NetworkLayer layer;
            // layer's weights updates
            double[][] layerWeightsUpdates;
            // layer's thresholds updates
            double[] layerThresholdUpdates;
            // neuron's weights updates
            double[] neuronWeightUpdates;

            // for each layer of the network
            for (int i = 0; i < network.Layers.Length; i++)
            {
                layer = network.Layers[i];
                layerWeightsUpdates = weightsUpdates[i];
                layerThresholdUpdates = thresholdsUpdates[i];

                // for each neuron of the layer
                for (int j = 0; j < layer.Neurons.Length; j++)
                {
                    neuron = layer.Neurons[j] as Neuron;
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
        ~Backpropagation()
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
                if (neuronErrors != null)
                {
                    neuronErrors.Dispose();
                    neuronErrors = null;
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
