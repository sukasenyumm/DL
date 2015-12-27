using System;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Collections.Generic;
using LibDL.Interface;
using LibDL.Utils;
using LibDL.ActivationFunction;

namespace LibDL
{
    [Serializable]
    public class DBN : NeuralNetwork
    {
        // DBN terdiri dari tumpukan jaringan RBM, disini digunakan generic tipe RBM untuk membuat stack
        private List<RBM> stackedRBM;
        
        // cara mendapatkan RBM pada setiap layer pada Deep Network
        public IList<RBM> StackedRBM
        {
            get { return stackedRBM; }
        }

        // cara mendapatkan banyaknya neuron output pada jaringan (ukuran dari vektor output yang telah dihitung)
        public int OutputCount
        {
            get
            {
                return (stackedRBM.Count == 0)?0:stackedRBM[stackedRBM.Count - 1].Hidden.Neurons.Length;
            }
        }

        public List<double> errCost;
        public string timeLearn;
        /*
         * Konstruktor dimana pembentukan RBM ada didalam class DBN
         */ 
        public DBN(int inputCount, params int[] hiddenNeurons) 
            : this(new BernoulliDistribution(alpha: 1), inputCount, hiddenNeurons) { }

        public DBN(IActivationFunction function, int inputCount, params int[] hiddenNeurons)
            : base(function,inputCount,hiddenNeurons)
        {
            stackedRBM = new List<RBM>();

            //buat layer pertama
            stackedRBM.Add(new RBM(
                hidden: new StochasticNetworkLayer(hiddenNeurons[0],inputCount,function),
                visible: new StochasticNetworkLayer(inputCount,hiddenNeurons[0],function)
                ));

            //buat layer yang lain
            for(int i=1;i<hiddenNeurons.Length;i++)
            {
                stackedRBM.Add(new RBM(
                hidden: new StochasticNetworkLayer(hiddenNeurons[i], hiddenNeurons[i-1], function),
                visible: new StochasticNetworkLayer(hiddenNeurons[i - 1], hiddenNeurons[i], function)
                ));
            }

            //override Layer from base
            //slayers = new StochasticNetworkLayer[stackedRBM.Count];
            //for (int i = 0; i < slayers.Length; i++)
            //    slayers[i] = stackedRBM[i].Hidden;

            layers = new NetworkLayer[stackedRBM.Count];
            for (int i = 0; i < layers.Length; i++)
                layers[i] = stackedRBM[i].Hidden;
        }

        /*
         * Konstruktor dimana pembentukan RBM ada diluar class DBN
         */ 
        public DBN(int inputCount, params RBM[] RBMlayers)
            : base(null, inputCount, new int[RBMlayers.Length])
        {
            stackedRBM = new List<RBM>(RBMlayers);

            ////override Layer from base
            //base.slayers = new StochasticNetworkLayer[stackedRBM.Count];
            //for (int i = 0; i < RBMlayers.Length; i++)
            //    base.slayers[i] = stackedRBM[i].Hidden;

            // Override AForge layers
            base.layers = new NetworkLayer[stackedRBM.Count];
            for (int i = 0; i < layers.Length; i++)
                base.layers[i] = stackedRBM[i].Hidden;
        }

        /*
         * Dapatkan output dari setiap RBM yang ditumpuk dari suatu nilai input
         * p(v_i = 1)=\sigma*(\Sigma(h_j*w_ij)*bias)
         * dimana \sigma adalah fungsi logistic \sigma(x) = 1/(1+exp(-x))
         * untuk setiap layer RBM yang ditumpuk,
         */
        public override double[] Compute(double[] input)
        {
            double[] output = input;

            foreach (RBM stackedLayer in stackedRBM)
                output = stackedLayer.Hidden.Compute(output);

            return output;
        }

        public double[] ZeroOutput(double[] output)
        {
            double[] outp=output;
            for(int i=0;i<output.Length;i++)
            {
                if(output[i]>0.5 && output[i]<=1.0)
                {
                    output[i] = 1.0;
                }
                else
                {
                    output[i] = 0.0;
                }
            }
            return outp;
        }

        /*
         * Dapatkan output dari setiap RBM yang ditumpuk tetapi pada
         * batasan tumpukan layer tertentu (range) dari suatu nilai input
         * 
         */
        public double[] Compute(double[] input, int layerIndex)
        {
            double[] output = input;

            for (int i = 0; i <= layerIndex;i++ )
                output = stackedRBM[i].Hidden.Compute(output);

            return output;
        }

        /*
         * 
         * Proses rekonstruksi:
         * p(h_j = 1)=\sigma*(\Sigma(v_i*w_ij)*bias)
         * dimana \sigma adalah fungsi logistic \sigma(x) = 1/(1+exp(-x))
         * untuk setiap layer RBM yang ditumpuk,
         * mengembalikan probabilitas vektor input yang mungkin sudah dihitung dan diberikan dalam nilai output.
         * 
         */
        public double[] Reconstruct(double[] output, int layerIndex)
        {
            double[] input = output;

            for (int i = layerIndex; i >= 0; i--)
                input = stackedRBM[i].Visible.Compute(input);

            return input;
        }

        /*
        * Dapatkan output dari setiap RBM yang ditumpuk tetapi pada
        * batasan tumpukan layer tertentu (range) dari nilai output
        * 
        */
        public double[] Reconstruct(double[] output)
        {
            double[] input = output;

            for (int i = stackedRBM.Count - 1; i >= 0; i--)
                input = stackedRBM[i].Visible.Compute(input);

            return input;
        }

        public void UpdateVisibleWeights()
        {
            foreach (var machine in stackedRBM)
                machine.UpdateVisibleWeights();
        }

        public void setAllErrCost(List<double> allcosterr)
        {
            errCost = new List<double>();
            errCost = allcosterr;
        }

        public List<double> getAllErrCost()
        {
            return errCost;
        }

        public void setTimeLearn(string time)
        {
            timeLearn = time;
        }

        public string getTimeLearn()
        {
            return timeLearn;
        }

        /// <summary>
        ///   Saves the network to a stream.
        /// </summary>
        /// 
        /// <param name="stream">The stream to which the network is to be serialized.</param>
        /// 
        public new void Save(Stream stream)
        {
            BinaryFormatter b = new BinaryFormatter();
            b.Serialize(stream, this);
        }

        /// <summary>
        ///   Saves the network to a stream.
        /// </summary>
        /// 
        /// <param name="path">The file path to which the network is to be serialized.</param>
        /// 
        public new void Save(string path)
        {
            //base.Save(@"D:\dataNN.bin");
            using (FileStream fs = new FileStream(path,FileMode.Create, FileAccess.Write, FileShare.None))
            {
                Save(fs);
            }
        }

        /// <summary>
        ///   Loads a network from a stream.
        /// </summary>
        /// 
        /// <param name="stream">The network from which the machine is to be deserialized.</param>
        /// 
        /// <returns>The deserialized network.</returns>
        /// 
        public static new DBN Load(Stream stream)
        {
            BinaryFormatter b = new BinaryFormatter();
            return (DBN)b.Deserialize(stream);
        }

        /// <summary>
        ///   Loads a network from a file.
        /// </summary>
        /// 
        /// <param name="path">The path to the file from which the network is to be deserialized.</param>
        /// 
        /// <returns>The deserialized network.</returns>
        /// 
        public static new DBN Load(string path)
        {
            using (FileStream fs = new FileStream(path, FileMode.Open))
            {
                return Load(fs);
            }
        }

    }
}
