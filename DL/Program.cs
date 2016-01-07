#define DEBUG_RECONSTRUCTED
#undef DEBUG_RECONSTRUCTED
#define USING_ACCORD
#undef USING_ACCORD
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LibDL;
using LibDL.Utils;
using LibDL.Interface;
using LibDL.ActivationFunction;
using LibDL.Learning;
using Accord.Math;
using Accord.Statistics.Analysis;
using System.Diagnostics;
using System.IO;

#if USING_ACCORD
using Accord.Neuro;
using Accord.Neuro.ActivationFunctions;
using Accord.Neuro.Networks;
using Accord.Neuro.Learning;
using AForge.Neuro.Learning;
using AForge.Neuro;
#endif

namespace DL
{
    class Program
    {
        public static TimeSpan procTime = new TimeSpan();
        public static DBN DBNNet;
        public static NeuralNetwork NNet;
        public static List<double> errCostFunc = new List<double>();
        static void Main(string[] args)
        {
            Console.WriteLine("Selamat Datang.."); 
            Console.WriteLine("Untuk mode penyimpanan ketik 'true' \ndan jika load file bin ketik 'false'(prefer)..");
            Console.Write("Mode: ");
            bool devMode = bool.Parse(Console.ReadLine());
            ReadCSVData data = new ReadCSVData();
            string file;
            string skip = "lala";
            string fileModel = "";
            int menu=1;
            
            if (devMode)
            {
                Console.WriteLine("Harap tunggu sebentar");
                data.LoadData(@"D:\mnist_test.csv", true, false);
                data.Save(@"D:\data_mnist_test.bin");
            }
            else
            {
                Console.WriteLine("Jika sudah punya file model skip proses learning dengan ketik 'skip'? ");
                skip = Console.ReadLine();
                if(skip == "skip")
                {
                    Console.WriteLine("Ketikkan direktori file model: ");
                    fileModel = @Console.ReadLine();
                }
             
                Console.WriteLine("Pilih menu berikut: \n1. Data MNIST TRAINING 784\n2. Data HURUF\n3. Data MNIST TRAINING 400");
                Console.Write("Pilih Menu: ");
                menu = int.Parse(Console.ReadLine());
                if (menu == 1)
                {
                    file = @"D:\train300.bin";
                }
                else if (menu == 2)
                {
                    file = @"D:\dataHuruf.bin";
                }
                else if (menu == 3)
                {
                    file = @"D:\data_mnist_train.bin"; //for test just
                }
                else
                {
                    file = @"D:\data_mnist_train.bin";
                }
                data = ReadCSVData.Load(file);
                       
            }

            double[][] inputs;
            if (menu != 3)
                inputs = data.Inputs.ToArray();
            else
                inputs = data.WidthNormalization(data.Inputs.ToArray());
            double[][] outputs = data.Outputs.ToArray();

            string fileModelSimpan="";
            //DBN-rprop
            if (skip != "skip")
            {
                Console.Write("Simpan model dengan nama: ");
                fileModelSimpan = Console.ReadLine();
                string begin = DateTime.Now.ToLongTimeString();

                Console.WriteLine("Apakah ingin membuat jaringan DBN-DNN? (true=iya/false=tidak-ANN biasa)");
                bool isDBN = bool.Parse(Console.ReadLine());
                Console.WriteLine("Apakah ingin mematikan komputer saat selesai training?  (true=iya/false=tidak))");
                bool isSD = bool.Parse(Console.ReadLine());
                if (isDBN)
                {
                    Console.WriteLine("Isikan parameter DBN-DNN berikut");
                    Console.WriteLine("======Fase Pretraining====");
                    Console.Write("Masukkan Epoch=");
                    int epoch = int.Parse(Console.ReadLine());
                    Console.Write("Masukkan Ukuran Mini Batch=");
                    int mb = int.Parse(Console.ReadLine());
                    Console.Write("Masukkan Learning rate=");
                    double lr = double.Parse(Console.ReadLine());
                    Console.Write("Masukkan Momentum=");
                    double mm = double.Parse(Console.ReadLine());
                    Console.Write("Masukkan Weight Decay=");
                    double dc = double.Parse(Console.ReadLine());
                    Console.WriteLine("======Fase Training atau Fine Tuning====");
                    Console.Write("Masukkan Epoch=");
                    int epoch2 = int.Parse(Console.ReadLine());
                    Console.Write("Pilih true untuk RProp dan false untuk Backprop=");
                    bool isRprop = bool.Parse(Console.ReadLine());
                    Console.Write("Masukkan Ukuran Mini Batch=");
                    int mb2 = int.Parse(Console.ReadLine());
                    Console.Write("Masukkan Learning rate=");
                    double lr2 = double.Parse(Console.ReadLine());
                    Console.Write("Masukkan Momentum=");
                    double mm2 = double.Parse(Console.ReadLine());

                    CreateNetwork(inputs.First().Length, outputs.First().Length);
                    //PreTraining(DBNNet, inputs, epoch, mb, 0.1, 0.5, 0.001, fileModelSimpan);
                    PreTraining(DBNNet, inputs, epoch, mb, lr, mm, dc, fileModelSimpan);
                    //Training(DBNNet, inputs, outputs, epoch2, 100, 0.5, 0.5, fileModelSimpan, false);

                    //DBNNet = DBN.Load(@"D:\embuhRBM_2");
                    //DBNNet.SetActivationFunction(new LibDL.ActivationFunction.SigmoidFunction(0.01));
                    Training(DBNNet, inputs, outputs, epoch2, mb2, lr2, mm2, fileModelSimpan, isRprop);

                    DBNNet.setTimeLearn(begin);
                    DBNNet.setAllErrCost(errCostFunc);
                    DBNNet.Save(@fileModelSimpan + "final");
                    CreateLogValidation(DBNNet, inputs, outputs, data, false);

                }
                else
                {
                    Console.WriteLine("======Fase Training ANN DNN/MLP====");
                    Console.Write("Masukkan Epoch=");
                    int epoch2 = int.Parse(Console.ReadLine());
                    Console.Write("Pilih true untuk RProp dan false untuk Backprop=");
                    bool isRprop = bool.Parse(Console.ReadLine());
                    Console.Write("Masukkan Ukuran Mini Batch=");
                    int mb2 = int.Parse(Console.ReadLine());
                    Console.Write("Masukkan Learning rate=");
                    double lr2 = double.Parse(Console.ReadLine());
                    Console.Write("Masukkan Momentum=");
                    double mm2 = double.Parse(Console.ReadLine());

                    CreateNetwork(inputs.First().Length, outputs.First().Length,false);
                    Training(NNet, inputs, outputs, epoch2, mb2, lr2, mm2, fileModelSimpan, isRprop);

                    NNet.setTimeLearn(begin);
                    NNet.setAllErrCost(errCostFunc);
                    NNet.Save(@fileModelSimpan + "final");
                    CreateLogValidation(NNet, inputs, outputs, data, false);
                }

                if (isSD)
                {
                    var psi = new ProcessStartInfo("shutdown", "/s /t 0");
                    psi.CreateNoWindow = true;
                    psi.UseShellExecute = false;
                    Process.Start(psi);
                }
            }
            else
            {
                NeuralNetwork network = NeuralNetwork.Load(fileModel);
                CreateLogValidation(network, inputs, outputs, data, false);
            }
           
        }
#if USING_ACCORD
        public static void CreateLogValidation(ActivationNetwork network, double[][] inputs, double[][] outputs, ReadCSVData data, bool isHuruf)
        {
            int[] actual = new int[outputs.Length];
            int[] expected = new int[outputs.Length];


            for (int i = 0; i < outputs.Length; i++)
            {
                actual[i] = data.ClassOutput(outputs[i], outputs.First().Length);
                if (isHuruf)
                {
                    if (actual[i] == 26)
                        actual[i] = 0;
                }
                else
                {
                    if (actual[i] == 10)
                        actual[i] = 0;
                }
            }

            int cou = 0;
            for (int i = 0; i < outputs.Length; i++)
            {
                expected[i] = data.ClassOutput(network.Compute(inputs[i]), outputs.First().Length);
                if (isHuruf)
                {
                    if (expected[i] == 26)
                    {
                        cou++;
                        expected[i] = 0;
                    }
                }
                else
                {
                    if (expected[i] == 10)
                    {
                        cou++;
                        expected[i] = 0;
                    }
                }
            }

            int numclassses = outputs.First().Length;
            if (isHuruf)
                numclassses = outputs.First().Length - 1;
            GeneralConfusionMatrix validation = new GeneralConfusionMatrix(numclassses, actual, expected);

            using (StreamWriter w = File.AppendText(@"D:\logAnalysis.txt"))
            {
                Log(validation, "debug", outputs, isHuruf, w);
                //Log("Test2", w);
            }

            using (StreamReader r = File.OpenText(@"D:\logAnalysis.txt"))
            {
                DumpLog(r);
            }
        }
#endif
        public static void CreateLogValidation(DBN network,double[][] inputs,double[][] outputs,ReadCSVData data,bool isHuruf)
        {
            int[] actual = new int[outputs.Length];
            int[] expected = new int[outputs.Length];


            for (int i = 0; i < outputs.Length; i++)
            {
                actual[i] = data.ClassOutput(outputs[i], outputs.First().Length);
                if (isHuruf)
                {
                    if (actual[i] == 26)
                        actual[i] = 0;
                }
                else
                {
                    if (actual[i] == 10)
                        actual[i] = 0;
                }
            }

            int cou = 0;
            for (int i = 0; i < outputs.Length; i++)
            {
                expected[i] = data.ClassOutput(network.ZeroOutput(network.Compute(inputs[i])), outputs.First().Length);
                if (isHuruf)
                {
                    if (expected[i] == 26)
                    {
                        cou++;
                        expected[i] = 0;
                    }
                }
                else
                {
                    if (expected[i] == 10)
                    {
                        cou++;
                        expected[i] = 0;
                    }
                }
            }

            int numclassses = outputs.First().Length;
            if (isHuruf)
                numclassses = outputs.First().Length - 1;
            GeneralConfusionMatrix validation = new GeneralConfusionMatrix(numclassses, actual, expected);

            using (StreamWriter w = File.AppendText(@"D:\logAnalysis.txt"))
            {
                Log(validation, network.getTimeLearn(), outputs, isHuruf, w);
                //Log("Test2", w);
            }

            using (StreamReader r = File.OpenText(@"D:\logAnalysis.txt"))
            {
                DumpLog(r);
            }
        }
        public static void CreateLogValidation(NeuralNetwork network, double[][] inputs, double[][] outputs, ReadCSVData data, bool isHuruf)
        {
            int[] actual = new int[outputs.Length];
            int[] expected = new int[outputs.Length];


            for (int i = 0; i < outputs.Length; i++)
            {
                actual[i] = data.ClassOutput(outputs[i], outputs.First().Length);
                if (isHuruf)
                {
                    if (actual[i] == 26)
                        actual[i] = 0;
                }
                else
                {
                    if (actual[i] == 10)
                        actual[i] = 0;
                }
            }

            int cou = 0;
            for (int i = 0; i < outputs.Length; i++)
            {
                expected[i] = data.ClassOutput(network.Compute(inputs[i]), outputs.First().Length);
                if (isHuruf)
                {
                    if (expected[i] == 26)
                    {
                        cou++;
                        expected[i] = 0;
                    }
                }
                else
                {
                    if (expected[i] == 10)
                    {
                        cou++;
                        expected[i] = 0;
                    }
                }
            }

            int numclassses = outputs.First().Length;
            if (isHuruf)
                numclassses = outputs.First().Length - 1;
            GeneralConfusionMatrix validation = new GeneralConfusionMatrix(numclassses, actual, expected);

            using (StreamWriter w = File.AppendText(@"D:\logAnalysis.txt"))
            {
                Log(validation, network.getTimeLearn(), outputs, isHuruf, w);
                //Log("Test2", w);
            }

            using (StreamReader r = File.OpenText(@"D:\logAnalysis.txt"))
            {
                DumpLog(r);
            }
        }

        public static void Log(GeneralConfusionMatrix validation,string time,double[][] outputs, bool isHuruf, TextWriter w)
        {
            w.Write("\r\nLog Entry : ");
            w.WriteLine("{0} {1}", DateTime.Now.ToLongTimeString(),
                DateTime.Now.ToLongDateString());
            w.WriteLine("Learning time occured:" + time);
            w.WriteLine("============================================================================================");
            w.WriteLine("Number of Sample:" + validation.Samples);
            w.WriteLine("Accuracy:" + validation.OverallAgreement);
            w.WriteLine("Geometric Agreement:" + validation.GeometricAgreement);
            w.WriteLine("Chance Agreement:" + validation.ChanceAgreement);
            w.WriteLine("Kappa:" + validation.Kappa);
            w.WriteLine("Confusion Matrix:");

            int numOut = outputs.First().Length;
            if(isHuruf)
                numOut= outputs.First().Length-1;

            for (int j = 0; j < numOut; j++)
            {
                w.Write("{0,-5},", j);
            }
            w.WriteLine();
            w.WriteLine();
            for (int i = 0; i < numOut; i++)
            {
                for (int j = 0; j < numOut; j++)
                {
                    w.Write("{0,-5},", validation.Matrix[i, j]);
                }
                w.WriteLine();
            }
            w.WriteLine("============================================================================================");
        }

        public static void DumpLog(StreamReader r)
        {
            string line;
            while ((line = r.ReadLine()) != null)
            {
                Console.WriteLine(line);
            }
        }

        public static int[] Random(int size, int groups)
        {
            // Create the index vector
            int[] idx = new int[size];

            if (groups == 1)
            {
                for (int i = 0; i < idx.Length; i++)
                    idx[i] = 0;
                return idx;
            }

            double n = groups / (double)size;
            for (int i = 0; i < idx.Length; i++)
                idx[i] = (int)System.Math.Ceiling((i + 0.9) * n) - 1;

            // Shuffle the indices vector
            Helper.Shuffle(idx);

            return idx;
        }

        public static void CreateNetwork(int inputLayer,int outputLayer,bool isDBN=true)
        {
            if(isDBN)
            {
                Console.WriteLine("Konfigurasi DBN-DNN");

                Console.Write("Pilih 2 jika dua hidden layer atau 3 jika 3 hidden layer= ");
                int hl = int.Parse(Console.ReadLine());
                DBN network;
                if (hl == 2)
                {
                    Console.Write("Berapa banyak hidden layer-1? ");
                    int hl1 = int.Parse(Console.ReadLine());
                    Console.Write("Berapa banyak hidden layer-2? ");
                    int hl2 = int.Parse(Console.ReadLine());
                    network = new DBN(new BernoulliDistribution(0.1),inputLayer, hl1, hl2, outputLayer);
                }
                else
                {
                    Console.Write("Berapa banyak hidden layer-1? ");
                    int hl1 = int.Parse(Console.ReadLine());
                    Console.Write("Berapa banyak hidden layer-2? ");
                    int hl2 = int.Parse(Console.ReadLine());
                    Console.Write("Berapa banyak hidden layer-3? ");
                    int hl3 = int.Parse(Console.ReadLine());
                    network = new DBN(new BernoulliDistribution(0.1),inputLayer, hl1, hl2, hl3, outputLayer);
                }
                DBNNet = network;            
            }
            else
            {
                Console.WriteLine("Konfigurasi ANN");

                Console.Write("Pilih 2 jika dua hidden layer atau 3 jika 3 hidden layer= ");
                int hl = int.Parse(Console.ReadLine());
                NeuralNetwork network;
                if (hl == 2)
                {
                    Console.Write("Berapa banyak hidden layer-1? ");
                    int hl1 = int.Parse(Console.ReadLine());
                    Console.Write("Berapa banyak hidden layer-2? ");
                    int hl2 = int.Parse(Console.ReadLine());
                    network = new NeuralNetwork(
                    new LibDL.ActivationFunction.SigmoidFunction(0.1),
                    inputLayer, 
                    hl1, 
                    hl2,
                    outputLayer); 
                }
                else
                {
                    Console.Write("Berapa banyak hidden layer-1? ");
                    int hl1 = int.Parse(Console.ReadLine());
                    Console.Write("Berapa banyak hidden layer-2? ");
                    int hl2 = int.Parse(Console.ReadLine());
                    Console.Write("Berapa banyak hidden layer-3? ");
                    int hl3 = int.Parse(Console.ReadLine());
                    network = new NeuralNetwork(
                    new LibDL.ActivationFunction.SigmoidFunction(0.1),
                    inputLayer,
                    hl1,
                    hl2,
                    hl3,
                    outputLayer); 
                }
                NNet = network;           
            }
        }

        public static void PreTraining(DBN network, double[][] inputs, int epoch, int batchSize, double lr,double mm,double dc,string @fileModelSimpan)
        {
            Console.WriteLine("==========PRE-TRAINING IN PROGRESS================");
            
            DBNLearning teacher = new DBNLearning(network)
            {
                LearningRate = lr,//0.1,
                Momentum = mm,//0.5,
                WeightDecay = dc,//0.001,
            };
            teacher.CreateCD();


            //learn 5000 iterations
            //Setup batches of input for learning.
            int batchCount = Math.Max(1, inputs.Length / batchSize);
            // Create mini-batches to speed learning.
            int[] groups = Random(inputs.Length, batchCount);
            double[][][] batches = inputs.Subgroups(groups);
            // Learning data for the specified layer.
            double[][][] layerData;

            var cd = teacher.GetLayerAlgorithm(teacher.LayerIndex);

            // Unsupervised learning on each hidden layer, except for the output layer.
            for (int layerIndex = 0; layerIndex < network.StackedRBM.Count - 1; layerIndex++)
            {
                teacher.LayerIndex = layerIndex;
                layerData = teacher.GetLayerInput(batches);
                for (int i = 0; i < epoch; i++)
                {
                    double error = teacher.RunEpoch(layerData) / inputs.Length;
                    errCostFunc.Add(error);
                    if (i % 10 == 0)
                    {
                        Console.WriteLine(i + ", Error = " + error);

                    }
                    //Breakpoint here to debug
#if DEBUG_RECONSTRUCTED
                    ReconstructImage.Delete(@"D:\reconstruct");
#endif
                }
                network.Save(@fileModelSimpan+"RBM_"+(layerIndex+1));
            }

            //new LibDL.Utils.GaussianWeights(network).Randomize();
            network.UpdateVisibleWeights();
            
        }

        public static void Training(NeuralNetwork network, double[][] inputs, double[][] outputs,int epoch, int batchSize, double lr, double mm, string @fileModelSimpan, bool useRprop = true)
        {
            Console.WriteLine("==========TRAINING IN PROGRESS================");
            //network.SetActivationFunction(new LibDL.ActivationFunction.SigmoidFunction(0.01));
            if(useRprop)
            {
                var teacher3 = new ResilientBackPropagationFineTuning(network);

                for (int i = 0; i < epoch/*200*/; i++)
                {
                    double error = teacher3.RunEpoch(inputs, outputs) / inputs.Length;
                    errCostFunc.Add(error);
                    if (i % 10 == 0)
                    {
                        Console.WriteLine(i + ", Error = " + error);

                    }
                }
            }
            else
            {
                var teacher3 = new Backpropagation(network)
                {
                    LearningRate = lr,//0.5,
                    Momentum = mm,//0.5,
                };


                //learn 5000 iterations
                //Setup batches of input for learning.
                int batchCountBP = Math.Max(1, inputs.Length / batchSize);
                // Create mini-batches to speed learning.
                int[] groupsBP = Random(inputs.Length, batchCountBP);
                double[][][] batchesIn = inputs.Subgroups(groupsBP);
                double[][][] batchesOut = outputs.Subgroups(groupsBP);
                // Learning data for the specified layer.
                double[][][] layerDataBPIn;
                double[][][] layerDataBPOut;
                layerDataBPIn = teacher3.GetLayerIO(batchesIn);
                layerDataBPOut = teacher3.GetLayerIO(batchesOut);


                for (int i = 0; i < epoch/*1000*/; i++)
                {
                    double error = (teacher3.RunEpoch(layerDataBPIn, layerDataBPOut) / layerDataBPIn.Length)/batchSize;
                    errCostFunc.Add(error);
                    if (i % 10 == 0)
                    {
                        Console.WriteLine(i + ", Error = " + error);

                    }
                }
            }
            network.Save(@fileModelSimpan + "FINAL");
        }
        
    }
}
