using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace MNISTCNTK
{
    class MNISTData
    {
        /// <summary>
        /// A class to convert the source MNIST data to CNTK readable txt file
        /// </summary>
        internal class MNISTDataConvertor
        {
            private const string train_data_image_path = "data/train-images.idx3-ubyte", train_data_label_path = "data/train-labels.idx1-ubyte";
            private const string test_data_image_path = "data/t10k-images.idx3-ubyte", test_data_label_path = "data/t10k-labels.idx1-ubyte";
            private readonly int train_data_count, test_data_count;
            private readonly List<byte[]> train_data_image = new List<byte[]>();
            private readonly List<byte> train_data_label = new List<byte>();
            private readonly List<byte[]> test_data_image = new List<byte[]>();
            private readonly List<byte> test_data_label = new List<byte>();

            public int Image_length { get; private set; } = 0;

            private enum MNISTDatabaseReadStatus
            {
                OK,
                ImageFileMagicNumberError,
                LabelFileMagicNumberError,
                ImageLableCountUnpairError,
                NoDataError,
            };

            public MNISTDataConvertor(int train_data_count = 60000, int test_data_count = 10000)
            {
                this.train_data_count = train_data_count;
                this.test_data_count = test_data_count;
            }

            public void ConvertAndSave(string path_train = "mnist-train.txt", string path_test = "mnist-test.txt")
            {
                // Read the data into memory
                ReadMNISTTrainData(train_data_count);
                ReadMNISTTestData(test_data_count);

                // Write them to text file
                using (FileStream train_text_stream = File.OpenWrite(path_train))
                {
                    StreamWriter train_writer = new StreamWriter(train_text_stream);

                    for (int i = 0; i < train_data_image.Count; i++)
                    {
                        byte[] image = train_data_image[i];
                        byte label = train_data_label[i];

                        string line = "|features ";

                        foreach (byte pixel in image)
                        {
                            line += $" {pixel}";
                        }

                        line += "\t|labels ";

                        for (int n = 0; n < 10; n++)
                        {
                            if (n == label)
                                line += " 1";
                            else
                                line += " 0";
                        }

                        train_writer.WriteLine(line);
                    }

                    train_writer.Close();
                }

                using (FileStream test_text_stream = File.OpenWrite(path_test))
                {
                    StreamWriter test_writer = new StreamWriter(test_text_stream);

                    for (int i = 0; i < test_data_image.Count; i++)
                    {
                        byte[] image = test_data_image[i];
                        byte label = test_data_label[i];

                        string line = "|features ";

                        foreach (byte pixel in image)
                        {
                            line += $" {pixel}";
                        }

                        line += "\t|labels ";

                        for (int n = 0; n < 10; n++)
                        {
                            if (n == label)
                                line += " 1";
                            else
                                line += " 0";
                        }

                        test_writer.WriteLine(line);
                    }

                    test_writer.Close();
                }
            }

            private MNISTDatabaseReadStatus ReadMNISTTrainData(int count)
            {
                FileStream image = File.OpenRead(train_data_image_path);
                FileStream label = File.OpenRead(train_data_label_path);

                BinaryReader image_rd = new BinaryReader(image);
                BinaryReader label_rd = new BinaryReader(label);

                // Check the magic number
                if (image_rd.ReadNonIntelInt32() != 0x00000803)
                {
                    return MNISTDatabaseReadStatus.ImageFileMagicNumberError;
                }

                if (label_rd.ReadNonIntelInt32() != 0x00000801)
                {
                    return MNISTDatabaseReadStatus.LabelFileMagicNumberError;
                }

                // Read the number of images and labels
                int i_count = image_rd.ReadNonIntelInt32();
                int l_count = label_rd.ReadNonIntelInt32();

                // Check the number of images and labels
                if (i_count != l_count)
                {
                    return MNISTDatabaseReadStatus.ImageLableCountUnpairError;
                }

                if (i_count <= 0)
                {
                    return MNISTDatabaseReadStatus.NoDataError;
                }

                // Read the resolution of image
                int image_w = image_rd.ReadNonIntelInt32();
                int image_h = image_rd.ReadNonIntelInt32();

                // Starting read
                for (int i = 0; i < i_count; i++)
                {
                    //Read the file
                    byte cLabel = label_rd.ReadByte();
                    //byte[,] cImage = new byte[image_w, image_h];
                    byte[] cImageFlat = new byte[image_w * image_h];
                    Image_length = image_w * image_h;

                    // Read the image
                    int j = 0;
                    for (int x = 0; x < image_w; x++)
                    {
                        for (int y = 0; y < image_h; y++, j++)
                        {
                            cImageFlat[j] = image_rd.ReadByte();
                            //cImage[x, y] = (byte)cImage[x, y];
                        }
                    }

                    // Generate IO set
                    train_data_image.Add(cImageFlat);
                    train_data_label.Add(cLabel);

                    if (count == i)
                    {
                        break;
                    }
                }

                image_rd.Close();
                label_rd.Close();

                return MNISTDatabaseReadStatus.OK;
            }

            private MNISTDatabaseReadStatus ReadMNISTTestData(int count)
            {
                FileStream image = File.OpenRead(test_data_image_path);
                FileStream label = File.OpenRead(test_data_label_path);

                BinaryReader image_rd = new BinaryReader(image);
                BinaryReader label_rd = new BinaryReader(label);

                // Check the magic number
                if (image_rd.ReadNonIntelInt32() != 0x00000803)
                {
                    return MNISTDatabaseReadStatus.ImageFileMagicNumberError;
                }

                if (label_rd.ReadNonIntelInt32() != 0x00000801)
                {
                    return MNISTDatabaseReadStatus.LabelFileMagicNumberError;
                }

                // Read the number of images and labels
                int i_count = image_rd.ReadNonIntelInt32();
                int l_count = label_rd.ReadNonIntelInt32();

                // Check the number of images and labels
                if (i_count != l_count)
                {
                    return MNISTDatabaseReadStatus.ImageLableCountUnpairError;
                }

                if (i_count <= 0)
                {
                    return MNISTDatabaseReadStatus.NoDataError;
                }

                // Read the resolution of image
                int image_w = image_rd.ReadNonIntelInt32();
                int image_h = image_rd.ReadNonIntelInt32();

                // Starting read
                for (int i = 0; i < i_count; i++)
                {
                    //Read the file
                    byte cLabel = label_rd.ReadByte();
                    //byte[,] cImage = new byte[image_w, image_h];
                    byte[] cImageFlat = new byte[image_w * image_h];

                    // Read the image
                    int j = 0;
                    for (int x = 0; x < image_w; x++)
                    {
                        for (int y = 0; y < image_h; y++, j++)
                        {
                            cImageFlat[j] = image_rd.ReadByte();
                            //cImage[x, y] = (byte)cImage[x, y];
                        }
                    }

                    // Generate IO set
                    test_data_image.Add(cImageFlat);
                    test_data_label.Add(cLabel);

                    if (count == i)
                    {
                        break;
                    }
                }

                image_rd.Close();
                label_rd.Close();

                return MNISTDatabaseReadStatus.OK;
            }
        }
    }

    public static class BigEndianUtils
    {
        public static int ReadNonIntelInt32(this BinaryReader br)
        {
            byte[] bytes = br.ReadBytes(sizeof(int));
            if (BitConverter.IsLittleEndian)
            {
                Array.Reverse(bytes);
            }

            return BitConverter.ToInt32(bytes, 0);
        }
    }
}
