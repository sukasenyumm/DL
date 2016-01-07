using System;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Collections.Generic;

namespace LibDL.Utils
{
    [Serializable]
    public class ReconstructImage
    {
        public List<double[]> reconstruct=new List<double[]>();
        public ReconstructImage()
        {

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

        public static void Delete(string path)
        {
            if (System.IO.File.Exists(path))
                try { System.IO.File.Delete(path); }
                finally { }
        }

       
        public static ReconstructImage Load(Stream stream)
        {
            BinaryFormatter b = new BinaryFormatter();
            return (ReconstructImage)b.Deserialize(stream);
        }

        public static ReconstructImage Load(string path)
        {
            using (FileStream fs = new FileStream(path, FileMode.Open))
            {
                return Load(fs);
            }
        }
    }
}
