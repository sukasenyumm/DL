#define DEBUG_RECONSTRUCTED
#undef DEBUG_RECONSTRUCTED
using System;
using LibDL.Interface;
using LibDL.Utils;
using System.Threading;
using System.Threading.Tasks;
using LibDL.ActivationFunction;

namespace LibDL.Learning
{
    public class ContrastiveDivergence : IDisposable
    {
        //tambahan parameter momentum pada update gradien decent
        private double momentum = 0.9;
        public double Momentum
        {
            get { return momentum; }
            set { momentum = value; }
        }
        //learning rate pada proses pelatihan (update gradien decent)
        private double learningRate = 0.1;
        public double LearningRate
        {
            get { return learningRate; }
            set { learningRate = value; }
        }
        //weigt decay untuk tambahan pada gradien, Istilah tambahan turunan dari fungsi yang melakukan pinalize 
        //"menghukum" weight yg tertalu besar
        private double weightDecay = 0.01;
        public double WeightDecay
        {
            get { return weightDecay; }
            set { weightDecay = value; }
        }
        //step pada gibbs sampling
        
        //parameter untuk membentuk bobot pada gradien dan membentuk output product dari positive gradien dan negative gradien
        //yang dihasilkan dari visible dan hidden unit
        private double[][] weightsGradient;
        private double[] vBiasGradient;
        private double[] hBiasGradient;


        //parameter update nilai pada bobot, output dari visible dan hidden
        private double[][] weightsUpdates;
        private double[] vBiasUpdates;
        private double[] hBiasUpdates;

        private int inputsCount;
        private int hiddenCount;

        private StochasticNetworkLayer hidden;
        private StochasticNetworkLayer visible;

        private ThreadLocal<ParallelStorage> storage;

#if DEBUG_RECONSTRUCTED
        public ReconstructImage Data = new ReconstructImage();
#endif
        public ContrastiveDivergence(RBM network)
        {
            init(network.Hidden, network.Visible);
        }

        public ContrastiveDivergence(StochasticNetworkLayer hidden, StochasticNetworkLayer visible)
        {
            init(hidden, visible);
        }

        /*
         * Inisialisasikan gradien 
         * 
         */ 
        private void init(StochasticNetworkLayer hidden, StochasticNetworkLayer visible)
        {
            //inisialisasi visible dan hidden unit
            this.hidden = hidden;
            this.visible = visible;

            //input count dari banyaknya neuron visible layer pada RBM
            inputsCount = hidden.InputsCount;
            //hidden count dari banykanya heuron pada hidden layer pada RBM
            hiddenCount = hidden.Neurons.Length;

            //inisialisasi bobot gradien matrik dengan ukuran visible x hidden
            weightsGradient = new double[inputsCount][];
            for (int i = 0; i < weightsGradient.Length; i++)
                weightsGradient[i] = new double[hiddenCount];

            //buat objek visible/hidden gradien
            vBiasGradient = new double[inputsCount];
            hBiasGradient = new double[hiddenCount];

            //inisialisasi bobot update matrik dengan ukuran matrik vsible x hidden
            weightsUpdates = new double[inputsCount][];
            for (int i = 0; i < weightsUpdates.Length; i++)
                weightsUpdates[i] = new double[hiddenCount];

            //buat objek update visible/hidden
            vBiasUpdates = new double[inputsCount];
            hBiasUpdates = new double[hiddenCount];

            //siapkan penyimpanan local pada class pararellstorage
            storage = new ThreadLocal<ParallelStorage>(() =>
                new ParallelStorage(inputsCount, hiddenCount));
        }

        /// <summary>
        ///   Not supported.
        /// </summary>
        /// 
        public double Run(double[] input)
        {
            throw new NotSupportedException();
        }
        public double RunEpoch(double[][] input)
        {
            // Initialize iteration karena training ini terjadi pada setiap perulangan/iterasi (epoch) maka nilai bobot gradien,visble gradien,
            // hidden gradien dari epoch sebelumnya harus dibersihkan terlebih dahulu nilai-nilainya.
            for (int i = 0; i < weightsGradient.Length; i++)
                Array.Clear(weightsGradient[i], 0, weightsGradient[i].Length);
            Array.Clear(hBiasGradient, 0, hBiasGradient.Length);
            Array.Clear(vBiasGradient, 0, vBiasGradient.Length);


            // HItung gradien dan erro pada model (error reconstruction dari layer)
            double error = ComputeGradient(input);

            // calculate weights updates
            CalculateUpdates(input);

            // update the network
            UpdateNetwork();
#if DEBUG_RECONSTRUCTED
            Data.Save(@"D:\reconstruct");
#endif
            return error;
        }

        private double ComputeGradient(double[][] input)
        {
            double errors = 0;
            

#if NET35
            var partial = storage.Value.Clear();
            for (int i = 0; i < input.Length; i++)
            {
                int observationIndex = i;
#else
            Object lockObj = new Object();

            // For each training instance
            Parallel.For(0, input.Length,

#if DEBUG
 new ParallelOptions() { MaxDegreeOfParallelism = Environment.ProcessorCount / 2 + Environment.ProcessorCount / 3 },
#endif

                // Initialize
                () => storage.Value.Clear(),

                // Map
                (observationIndex, loopState, partial) =>
#endif
                {
                    var observation = input[observationIndex];

                    var probability = partial.OriginalProbability;
                    var activations = partial.OriginalActivations;
                    var reconstruction = partial.ReconstructedInput;
                    var reprobability = partial.ReconstructedProbs;

                    var weightGradient = partial.WeightGradient;
                    var hiddenGradient = partial.HBiasGradient;
                    var visibleGradient = partial.VBiasGradient;


                    // for information: http://www.cs.toronto.edu/~hinton/code/rbm.m
                    // 1. Compute a forward pass. The network is being
                    //    driven by data, so we will gather activations
                    for (int j = 0; j < hidden.Neurons.Length; j++)
                    {
                        probability[j] = hidden.Neurons[j].Compute(observation);  // output probabilities
                        activations[j] = hidden.Neurons[j].Generate(probability[j]); // state activations ==> h0
                    }

                    // 2. Reconstruct inputs from previous outputs ==> v1
                    for (int j = 0; j < visible.Neurons.Length; j++)
                        reconstruction[j] = visible.Neurons[j].Compute(activations);

#if DEBUG_RECONSTRUCTED
                    Data.reconstruct.Add(reconstruction);
#endif

                    // 3. Compute outputs for the reconstruction. The network
                    //    is now being driven by reconstructions, so we should
                    //    gather the output probabilities without sampling ==> h1
                    for (int j = 0; j < hidden.Neurons.Length; j++)
                        reprobability[j] = hidden.Neurons[j].Compute(reconstruction);



                    // 4.1. Compute positive associations
                    for (int k = 0; k < observation.Length; k++)
                        for (int j = 0; j < probability.Length; j++)
                            weightGradient[k][j] += observation[k] * probability[j];

                    for (int j = 0; j < hiddenGradient.Length; j++)
                        hiddenGradient[j] += probability[j];

                    for (int j = 0; j < visibleGradient.Length; j++)
                        visibleGradient[j] += observation[j];


                    // 4.2. Compute negative associations
                    for (int k = 0; k < reconstruction.Length; k++)
                        for (int j = 0; j < reprobability.Length; j++)
                            weightGradient[k][j] -= reconstruction[k] * reprobability[j];

                    for (int j = 0; j < reprobability.Length; j++)
                        hiddenGradient[j] -= reprobability[j];

                    for (int j = 0; j < reconstruction.Length; j++)
                        visibleGradient[j] -= reconstruction[j];


                    // Compute current error
                    for (int j = 0; j < observation.Length; j++)
                    {
                        double e = observation[j] - reconstruction[j];
                        partial.ErrorSumOfSquares += e * e;
                    }

#if !NET35
                    return partial; // Report partial solution
                },

                // Reduce
                (partial) =>
                {
                    lock (lockObj)
                    {
                        // Accumulate partial solutions
                        for (int i = 0; i < weightsGradient.Length; i++)
                            for (int j = 0; j < weightsGradient[i].Length; j++)
                                weightsGradient[i][j] += partial.WeightGradient[i][j];

                        for (int i = 0; i < hBiasGradient.Length; i++)
                            hBiasGradient[i] += partial.HBiasGradient[i];

                        for (int i = 0; i < vBiasGradient.Length; i++)
                            vBiasGradient[i] += partial.VBiasGradient[i];

                        errors += partial.ErrorSumOfSquares;
                    }
                });
#else
                }
            }

            weightsGradient = partial.WeightGradient;
            hiddenBiasGradient = partial.HiddenBiasGradient;
            visibleBiasGradient = partial.VisibleBiasGradient;
            errors = partial.ErrorSumOfSquares;
#endif

            return errors;
        }

        private void CalculateUpdates(double[][] input)
        {
            double rate = learningRate;

            // Assume all neurons in the layer have the same act function
            rate = learningRate / (input.Length);


            // 5. Compute gradient descent updates
            for (int i = 0; i < weightsGradient.Length; i++)
                for (int j = 0; j < weightsGradient[i].Length; j++)
                    weightsUpdates[i][j] = momentum * weightsUpdates[i][j]
                        + (rate * weightsGradient[i][j]);

            for (int i = 0; i < hBiasUpdates.Length; i++)
                hBiasUpdates[i] = momentum * hBiasUpdates[i]
                        + (rate * hBiasGradient[i]);

            for (int i = 0; i < vBiasUpdates.Length; i++)
                vBiasUpdates[i] = momentum * vBiasUpdates[i]
                        + (rate * vBiasGradient[i]);

            //System.Diagnostics.Debug.Assert(!weightsGradient.HasNaN());
            //System.Diagnostics.Debug.Assert(!vBiasUpdates.HasNaN());
            //System.Diagnostics.Debug.Assert(!hBiasUpdates.HasNaN());
        }

        private void UpdateNetwork()
        {
            // 6.1 Update hidden layer weights
            for (int i = 0; i < hidden.Neurons.Length; i++)
            {
                StochasticNeuron neuron = hidden.Neurons[i];
                for (int j = 0; j < neuron.Weights.Length; j++)
                    neuron.Weights[j] += weightsUpdates[j][i] - learningRate * weightDecay * neuron.Weights[j];
                neuron.Threshold += hBiasUpdates[i];
            }

            // 6.2 Update visible layer with reverse weights
            for (int i = 0; i < visible.Neurons.Length; i++)
                visible.Neurons[i].Threshold += vBiasUpdates[i];
            visible.CopyReversedWeightsFrom(hidden);
        }

        private class ParallelStorage
        {
            public double[][] WeightGradient { get; set; }
            public double[] VBiasGradient { get; set; }
            public double[] HBiasGradient { get; set; }

            public double[] OriginalActivations { get; set; }
            public double[] OriginalProbability { get; set; }

            public double[] ReconstructedInput { get; set; }
            public double[] ReconstructedProbs { get; set; }

            public double ErrorSumOfSquares { get; set; }

            public ParallelStorage(int inputsCount, int hiddenCount)
            {
                WeightGradient = new double[inputsCount][];
                for (int i = 0; i < WeightGradient.Length; i++)
                    WeightGradient[i] = new double[hiddenCount];

                VBiasGradient = new double[inputsCount];
                HBiasGradient = new double[hiddenCount];

                OriginalActivations = new double[hiddenCount];
                OriginalProbability = new double[hiddenCount];
                ReconstructedInput = new double[inputsCount];
                ReconstructedProbs = new double[hiddenCount];
            }

            public ParallelStorage Clear()
            {
                ErrorSumOfSquares = 0;
                for (int i = 0; i < WeightGradient.Length; i++)
                    Array.Clear(WeightGradient[i], 0, WeightGradient[i].Length);
                Array.Clear(VBiasGradient, 0, VBiasGradient.Length);
                Array.Clear(HBiasGradient, 0, HBiasGradient.Length);
                return this;
            }

        }

        #region IDisposable members
        /// <summary>
        ///   Performs application-defined tasks associated with 
        ///   freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        /// 
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        ///   Releases unmanaged and - optionally - managed resources
        /// </summary>
        /// 
        /// <param name="disposing"><c>true</c> to release both managed and unmanaged
        /// resources; <c>false</c> to release only unmanaged resources.</param>
        /// 
        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                // free managed resources
                if (storage != null)
                {
                    storage.Dispose();
                    storage = null;
                }
            }
        }

        /// <summary>
        ///   Releases unmanaged resources and performs other cleanup operations before the
        ///   <see cref="ContrastiveDivergenceLearning"/> is reclaimed by garbage collection.
        /// </summary>
        /// 
        ~ContrastiveDivergence()
        {
            Dispose(false);
        }
        #endregion
    }
}
