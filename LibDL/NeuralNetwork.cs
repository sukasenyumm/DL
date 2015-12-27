using System;
using System.IO;
using System.Collections.Generic;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using LibDL.Interface;
using LibDL.Utils;
using LibDL.ActivationFunction;

namespace LibDL
{
    [Serializable]
    public class NeuralNetwork : INeuralNetwork
    {

        /*
         * Jumlah input data
         */
        private int inputsCount = 0;
        public int InputsCount
        {
            get { return inputsCount; }
            set { inputsCount = value; }
        }
        /*
         * Nilai output vektor dari layer
         */
        private double[] output;
   
        public double[] Output
        {
            get { return output; }
        }

        /*
         * Deklarasi beberapa activation neuron pada suatu layer
         */
        protected int layersCount = 0;
        protected NetworkLayer[] layers;

       
        public NetworkLayer[] Layers
        {
            get { return layers; }
        }

        protected UsualNetworkLayer[] usualLayers;


        public UsualNetworkLayer[] UsualLayers
        {
            get { return usualLayers; }
        }
        
        public List<double> errCost;
        public string timeLearn;

        public NeuralNetwork(IActivationFunction function, int inputsCount, params int[] neuronsCount)
        {
            this.inputsCount = Math.Max(1, inputsCount);
            this.layersCount = Math.Max(1, neuronsCount.Length);
          
         
            // create collection of layers
            this.layers = this.usualLayers = new UsualNetworkLayer[this.layersCount];
            for (int i = 0; i < usualLayers.Length; i++)
            {
                layers[i] = usualLayers[i] = new UsualNetworkLayer(
                    // neurons count in the layer
                    neuronsCount[i],
                    // inputs count of the layer
                    (i == 0) ? inputsCount : neuronsCount[i - 1],
                    // activation function of the layer
                    function);
            }
           
        }

        public virtual double[] Compute(double[] input)
        {
            // variabel lokal mencegah konfilk mutlithreading
            double[] output = input;

            
            for (int i = 0; i < usualLayers.Length; i++)
            {
                output = usualLayers[i].Compute(output);
            }
            

            // assign output
            this.output = output;

            return output;
        }

        //(KODE 1-2)
        public void Randomize()
        {
            foreach (NetworkLayer layer in layers)
            {
                layer.Randomize(); 
            }
        }

        public void SetActivationFunction(IActivationFunction function)
        {
            for (int i = 0; i < layers.Length; i++)
            {
                ((NetworkLayer)layers[i]).SetActivationFunction(function);
            }
        }

        /// <summary>
        /// Save network to specified file.
        /// </summary>
        /// 
        /// <param name="fileName">File name to save network into.</param>
        /// 
        /// <remarks><para>The neural network is saved using .NET serialization (binary formatter is used).</para></remarks>
        /// 
        public void Save(string fileName)
        {
            FileStream stream = new FileStream(fileName, FileMode.Create, FileAccess.Write, FileShare.None);
            Save(stream);
            stream.Close();
        }

        /// <summary>
        /// Save network to specified file.
        /// </summary>
        /// 
        /// <param name="stream">Stream to save network into.</param>
        /// 
        /// <remarks><para>The neural network is saved using .NET serialization (binary formatter is used).</para></remarks>
        /// 
        public void Save(Stream stream)
        {
            IFormatter formatter = new BinaryFormatter();
            formatter.Serialize(stream, this);
        }

        /// <summary>
        /// Load network from specified file.
        /// </summary>
        /// 
        /// <param name="fileName">File name to load network from.</param>
        /// 
        /// <returns>Returns instance of <see cref="Network"/> class with all properties initialized from file.</returns>
        /// 
        /// <remarks><para>Neural network is loaded from file using .NET serialization (binary formater is used).</para></remarks>
        /// 
        public static NeuralNetwork Load(string fileName)
        {
            FileStream stream = new FileStream(fileName, FileMode.Open, FileAccess.Read, FileShare.Read);
            NeuralNetwork network = Load(stream);
            stream.Close();

            return network;
        }

        /// <summary>
        /// Load network from specified file.
        /// </summary>
        /// 
        /// <param name="stream">Stream to load network from.</param>
        /// 
        /// <returns>Returns instance of <see cref="Network"/> class with all properties initialized from file.</returns>
        /// 
        /// <remarks><para>Neural network is loaded from file using .NET serialization (binary formater is used).</para></remarks>
        /// 
        public static NeuralNetwork Load(Stream stream)
        {
            IFormatter formatter = new BinaryFormatter();
            NeuralNetwork network = (NeuralNetwork)formatter.Deserialize(stream);
            return network;
        }


        public double[] ZeroOutput(double[] output)
        {
            throw new NotImplementedException();
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
    }
}
