using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LibDL.Interface;
using LibDL;

namespace LibDL.Learning
{
    public class DBNLearning
    {
        private DBN network;
        private ContrastiveDivergence[] algorithms;
        
        private int layerIndex = 0;

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
        public DBNLearning(DBN network)
        {
            this.network = network;
        }

        public void CreateCD()
        {
            algorithms = new ContrastiveDivergence[network.StackedRBM.Count];
            for (int i = 0; i < network.StackedRBM.Count; i++)
            {
                StochasticNetworkLayer hidden = network.StackedRBM[i].Hidden;
                StochasticNetworkLayer visible = network.StackedRBM[i].Visible;
                algorithms[i] = new ContrastiveDivergence(hidden, visible)
                {
                    LearningRate = this.learningRate,
                    Momentum = this.momentum,
                    WeightDecay = this.weightDecay,
                };
            }
        }

        public int LayerIndex
        {
            get { return layerIndex; }
            set
            {
                if (layerIndex < 0 || layerIndex >= network.StackedRBM.Count)
                    throw new ArgumentOutOfRangeException("value");

                layerIndex = value;
            }
        }

        public double[][] GetLayerInput(double[][] input)
        {
            return GetLayerInput(new[] { input })[0];
        }

        public double[][][] GetLayerInput(double[][][] batches)
        {
            if (layerIndex == 0)
                return batches;

            var outputBatches = new double[batches.Length][][];

            for (int j = 0; j < batches.Length; j++)
            {
                int batchSize = batches[j].Length;

                double[][] inputs = batches[j];
                double[][] outputs = new double[batchSize][];

                for (int i = 0; i < inputs.Length; i++)
                {
                    network.Compute(inputs[i]); // double[] responses = 
                    outputs[i] = network.StackedRBM[layerIndex - 1].Hidden.Output;
                }

                outputBatches[j] = outputs;
            }

            return outputBatches;
        }

        public ContrastiveDivergence GetLayerAlgorithm(int layerIndex)
        {
            return algorithms[layerIndex];
        }
        public double Run(double[] input)
        {
            // Get layer learning algorithm
            var teacher = algorithms[layerIndex];

            // Learn the layer using data
            return teacher.Run(input);
            
        }

        public double RunEpoch(double[][] input)
        {
            // Get layer learning algorithm
            var teacher = algorithms[layerIndex];

            // Learn the layer using data
            return teacher.RunEpoch(input);
        }

        public double RunEpoch(double[][][] batches)
        {
            // Get layer learning algorithm
            var teacher = algorithms[layerIndex];

            // Learn the layer using data
            double error = 0;
            for (int i = 0; i < batches.Length; i++)
                error += teacher.RunEpoch(batches[i]);

            return error;
        }

        public double ComputeError(double[][] inputs)
        {
            double error = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                double[] output = network.Compute(inputs[i]);
                double[] reconstruct = network.Reconstruct(output);

                for (int j = 0; j < inputs[i].Length; j++)
                {
                    double e = reconstruct[j] - inputs[i][j];
                    error += e * e;
                }
            }
            return error;
        }
    }
}
