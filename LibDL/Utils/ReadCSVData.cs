using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Text.RegularExpressions;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;

namespace LibDL.Utils
{
    [Serializable]
    public class ReadCSVData
    {
        private List<double[]> inputs = new List<double[]>();
        private List<double[]> outputs = new List<double[]>();

        public List<double[]> Inputs
        {
            get { return inputs; }
        }

        public List<double[]> Outputs
        {
            get { return outputs; }
        }

        public double[][] WidthNormalization(double[][] inputsMnist)
        {
            double[][] inputs = inputsMnist;

            double[][] inputstemp = new double[inputs.Length][];
            for (int i = 0; i < inputs.Length; ++i)
                inputstemp[i] = new double[400];//400 for 20x20 px

            for (int k = 0; k < inputs.Length; k++)
            {
                double[][] pixels = new double[28][];
                for (int i = 0; i < pixels.Length; ++i)
                    pixels[i] = new double[28];

                for (int i = 0; i < pixels.Length; i++)
                {
                    for (int j = 0; j < pixels.Length; j++)
                    {
                        pixels[i][j] = inputs[k][i * pixels.Length + j];
                    }
                }

                for (int i = 0; i < pixels.Length; ++i)
                {
                    pixels[i] = pixels[i].Skip(4).Take(24).ToArray(); //remove 4 pixel from left
                }

                for (int i = 0; i < pixels.Length; ++i)
                {
                    pixels[i] = pixels[i].Take(20).ToArray(); //remove 4 pixel from right
                }

                pixels = pixels.Skip(4).Take(24).ToArray(); //remove 4 pixel from up
                pixels = pixels.Take(20).ToArray(); //remove 4 pixel from bottom

                for (int x = 0; x < pixels.Length; x++)
                {
                    for (int y = 0; y < pixels.Length; y++)
                    {
                        inputstemp[k][x * pixels.Length + y] = pixels[x][y];
                    }
                }
            }

            return inputstemp;
        }
        private double[] OneToNEncoding(int constraint,int index,bool sign=false)
        {
            double[] outp = new double[constraint];
            for (int i = 0; i < constraint;i++ )
            {
                if ((constraint-index == constraint-1) && (i == constraint-1))
                    outp[i] = 1.0;
                else if (i == (constraint - index) && (constraint - index != constraint-1))
                    outp[i] = 1.0;
                else
                    outp[i] = (sign == true) ? -1.0 : 0.0;
            }
            return outp;
        }
        public int ClassOutput(double[] output,int nClasses)
        {
            return output.ToList().IndexOf(output.Max());
        }
        public void LoadData(string file,bool isDigit=false,bool sign=false)
        {

            try
            {
                var sourcePath = file; //HARDCODE
                var delimiter = ",";
                var firstLineContainsHeaders = false; //false!!
                var lineNumber = 0;

                var splitExpression = new Regex(@"(" + delimiter + @")(?=(?:[^""]|""[^""]*"")*$)");


                byte[][] pixels = new byte[28][];
                for (int i = 0; i < pixels.Length; ++i)
                    pixels[i] = new byte[28];

                //string[] temp = new string[745];

                using (var reader = new StreamReader(sourcePath))
                {
                    string line = null;
                    string[] headers = null;
                    if (firstLineContainsHeaders)
                    {
                        line = reader.ReadLine();
                        lineNumber++;

                        if (string.IsNullOrEmpty(line)) return; // file is empty;

                        headers = splitExpression.Split(line).Where(s => s != delimiter).ToArray();

                        //writer.WriteLine(line); // write the original header to the temp file.
                    }

                    //int count = 0;
                    //string sh = "";
                    while ((line = reader.ReadLine()) != null)
                    {
                        lineNumber++;

                        var columns = splitExpression.Split(line).Where(s => s != delimiter).ToArray();

                        // if there are no headers, do a simple sanity check to make sure you always have the same number of columns in a line
                        if (headers == null) headers = new string[columns.Length];

                        if (columns.Length != headers.Length) throw new InvalidOperationException(string.Format("Line {0} is missing one or more columns.", lineNumber));

                        //Console.WriteLine("Label huruf saat ini: '" + IntToLetters(int.Parse(columns[0])) + "'");
                        if (isDigit == false)
                        {
                            double[] arrInp = new double[784];
                            double[] arrOut = new double[26];
                        
                            bool isConvertible2 = false;
                            double t=0.0;
                            isConvertible2 = double.TryParse(columns[0], out t);
                            if (t == 0.0)
                                t = 26;
                            arrOut = OneToNEncoding(26, (int)t,sign);//26 for hurufToBin((int)t,8);//

                            bool isConvertible = false;

                        
                            for (int i = 1; i < 785; i++)
                            {
                                arrInp[i-1] = 0;
                                isConvertible = double.TryParse(columns[i], out arrInp[i-1]);
                            }
                            inputs.Add(arrInp);
                            outputs.Add(arrOut);
                        }
                        else
                        {
                            double[] arrInp = new double[784];
                            double[] arrOut = new double[26];

                            bool isConvertible2 = false;
                            double t = 0.0;
                            isConvertible2 = double.TryParse(columns[0], out t);
                            if (t == 0.0)
                                t = 10;
                            arrOut = OneToNEncoding(10, (int)t,sign);//26 for hurufToBin((int)t,8);//

                            bool isConvertible = false;

                            double tempo = 0.0;
                            for (int i = 1; i < 785; i++)
                            {
                                arrInp[i-1] = 0;
                                tempo = Convert.ToDouble(columns[i]);
                                if(sign)
                                    isConvertible = double.TryParse((tempo > 127.0) ? columns[i] = "1" : "-1", out arrInp[i - 1]);
                                else
                                    isConvertible = double.TryParse((tempo > 127.0) ? columns[i] = "1" : "0", out arrInp[i - 1]);
                            }
                            inputs.Add(arrInp);
                            outputs.Add(arrOut);
                        }           
                     
                    }

                }
                Console.WriteLine("Membaca data csv berhasil!!");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                Console.ReadLine();
            }
        }


        public void Save(Stream stream)
        {
            BinaryFormatter b = new BinaryFormatter();
            b.Serialize(stream, this);
        }

        public void Save(string path)
        {
            //base.Save(@"D:\dataNN.bin");
            using (FileStream fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None))
            {
                Save(fs);
            }
        }

        public static ReadCSVData Load(Stream stream)
        {
            BinaryFormatter b = new BinaryFormatter();
            return (ReadCSVData)b.Deserialize(stream);
        }

       
        public static ReadCSVData Load(string path)
        {
            using (FileStream fs = new FileStream(path, FileMode.Open))
            {
                return Load(fs);
            }
        }
    }
}
